"""Git-bus bridge: drive a live Colab session from `git push`.

Architecture
------------

Two directories in the repo act as a FIFO queue + log over git:

  bridge/queue/<task_id>.json   ← tasks pushed FROM Claude
  bridge/log/<task_id>.json     ← results pushed BACK by Colab

Colab runs ``run_bridge_loop()`` in a persistent cell. On each iteration:

  1. ``git fetch origin main`` + hard reset (picks up new tasks + code changes)
  2. List ``bridge/queue/*.json`` in lexicographic order; skip any ``task_id``
     that already has a ``bridge/log/<task_id>.json``.
  3. For each pending task, execute its spec and capture (stdout, stderr,
     return value, traceback). Write the log JSON.
  4. ``git add bridge/log/`` + commit + push.
  5. Sleep ``poll_seconds`` and loop.

Task spec format (one JSON file per task)::

    {
      "task_id": "20260519-235959-name",       # unique, sortable
      "kind":    "eval" | "call",
      "code":    "<python>",                    # for kind="eval"
      "module":  "src.sycophancy",              # for kind="call"
      "function": "run_sycophancy_probe",       # for kind="call"
      "args":    { ... },                       # for kind="call"
      "needs":   ["lm"],                        # globals to inject (optional)
      "save":    { "key": "results/<path>" }    # also dump these globals to disk
    }

For ``kind=eval`` the code runs in a namespace with whatever ``globals`` the
caller passed (typically ``{"lm": lm}``); any new keys created in the
namespace get returned to the log.

Authentication
--------------

To push back, the Colab runtime needs a GitHub token. Either set
``GH_TOKEN`` as a Colab secret or run ``setup_git_auth(token=...)`` once.
The bridge configures ``origin`` to the HTTPS URL with the token embedded.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = False, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["git", "-C", str(repo)] + list(args)
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def setup_git_auth(
    repo_dir: str,
    token: Optional[str] = None,
    remote: str = "origin",
    user_email: str = "colab-bridge@example.com",
    user_name: str = "colab-bridge",
) -> bool:
    """Configure the repo's ``origin`` to push as the token user.

    Reads ``GH_TOKEN`` from env if ``token`` is None. Returns True on success.
    """
    token = token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[bridge] GH_TOKEN not set — pushes will fail. Set Colab secret "
              "GH_TOKEN or pass token=... to setup_git_auth().")
        return False
    repo = Path(repo_dir)
    # Discover the remote owner/repo from the existing remote URL.
    cur = _git(repo, "remote", "get-url", remote, capture=True).stdout.strip()
    # Normalize SSH/HTTPS to a clean owner/repo path.
    if cur.startswith("git@github.com:"):
        path = cur.split(":", 1)[1]
    elif cur.startswith("https://"):
        path = cur.split("github.com/", 1)[-1]
    else:
        print(f"[bridge] cannot parse remote URL: {cur!r}")
        return False
    if path.endswith(".git"):
        path = path[:-4]
    new_url = f"https://x-access-token:{token}@github.com/{path}.git"
    _git(repo, "remote", "set-url", remote, new_url, check=True)
    _git(repo, "config", "user.email", user_email)
    _git(repo, "config", "user.name", user_name)
    print(f"[bridge] git auth configured for {path}")
    return True


def _pull_and_reset(repo: Path) -> None:
    _git(repo, "fetch", "origin", "main")
    _git(repo, "reset", "--hard", "origin/main")


def _commit_and_push(repo: Path, paths: list, message: str) -> bool:
    for p in paths:
        _git(repo, "add", str(p))
    diff = _git(repo, "diff", "--cached", "--quiet").returncode
    if diff == 0:
        return False  # nothing staged
    _git(repo, "commit", "-m", message)
    push = _git(repo, "push", capture=True)
    if push.returncode != 0:
        print(f"[bridge] push failed:\n{push.stderr}")
        return False
    return True


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------


def _execute_task(spec: Dict[str, Any], globals_inject: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one task spec; return a serialisable log payload."""
    kind = spec.get("kind", "eval")
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    payload: Dict[str, Any] = {
        "ok": False, "kind": kind,
        "stdout": "", "stderr": "", "return_value_repr": None,
        "traceback": None,
        "new_globals": [],
    }
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            if kind == "eval":
                code = spec.get("code", "")
                ns: Dict[str, Any] = dict(globals_inject)
                before = set(ns.keys())
                exec(code, ns)
                new_keys = [k for k in ns.keys() if k not in before]
                payload["new_globals"] = new_keys
                rv = ns.get("_result")
                if rv is not None:
                    payload["return_value_repr"] = repr(rv)[:8000]
            elif kind == "call":
                module = importlib.import_module(spec["module"])
                fn = getattr(module, spec["function"])
                args = spec.get("args", {})
                # Inject `lm` automatically when the function accepts it.
                import inspect
                sig = inspect.signature(fn)
                kwargs = dict(args)
                for name in sig.parameters:
                    if name in globals_inject and name not in kwargs:
                        kwargs[name] = globals_inject[name]
                rv = fn(**kwargs)
                payload["return_value_repr"] = repr(rv)[:8000]
            else:
                raise ValueError(f"unknown kind: {kind!r}")
        payload["ok"] = True
    except Exception:
        payload["traceback"] = traceback.format_exc()
    payload["stdout"] = stdout_buf.getvalue()[:30000]
    payload["stderr"] = stderr_buf.getvalue()[:8000]
    return payload


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _iter_pending(queue_dir: Path, log_dir: Path) -> Iterator[Path]:
    for task_file in sorted(queue_dir.glob("*.json")):
        if (log_dir / task_file.name).exists():
            continue
        yield task_file


def run_bridge_loop(
    repo_dir: str,
    globals_inject: Optional[Dict[str, Any]] = None,
    poll_seconds: int = 30,
    max_iters: Optional[int] = None,
    auto_push: bool = True,
    verbose: bool = True,
) -> None:
    """Persistent Colab loop. Run this in a single notebook cell.

    Parameters
    ----------
    repo_dir
        Path to the cloned repo (e.g. ``/content/diagnosticpercept``).
    globals_inject
        Dict of names made available to task code (typically ``{"lm": lm}``
        so tasks can use the already-loaded model without reloading weights).
    poll_seconds
        How often to ``git fetch`` for new tasks. Default 30s.
    max_iters
        Cap total iterations (useful for tests). ``None`` = forever.
    auto_push
        If True, ``git push`` each completed log JSON immediately.
    """
    repo = Path(repo_dir)
    queue = repo / "bridge" / "queue"
    log = repo / "bridge" / "log"
    queue.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    globals_inject = dict(globals_inject or {})

    iters = 0
    while max_iters is None or iters < max_iters:
        iters += 1
        try:
            _pull_and_reset(repo)
        except Exception as e:
            if verbose:
                print(f"[bridge] fetch error: {e}")
        pending = list(_iter_pending(queue, log))
        if verbose and pending:
            print(f"[bridge] {len(pending)} pending task(s)")
        for task_file in pending:
            spec = json.loads(task_file.read_text())
            task_id = spec.get("task_id") or task_file.stem
            if verbose:
                print(f"[bridge] executing {task_id} ...")
            t0 = time.time()
            result = _execute_task(spec, globals_inject)
            result["task_id"] = task_id
            result["wall_seconds"] = time.time() - t0
            log_path = log / task_file.name
            log_path.write_text(json.dumps(result, indent=2, default=str))
            if auto_push:
                _commit_and_push(repo, [log_path], f"bridge: {task_id} "
                                 f"{'ok' if result['ok'] else 'fail'}")
            if verbose:
                status = "ok" if result["ok"] else "fail"
                print(f"[bridge]   → {status} ({result['wall_seconds']:.1f}s)")
        time.sleep(poll_seconds)


# ---------------------------------------------------------------------------
# Convenience: pushing a task from local (used by helper scripts)
# ---------------------------------------------------------------------------


def push_task(
    repo_dir: str,
    task_id: str,
    kind: str = "eval",
    code: Optional[str] = None,
    module: Optional[str] = None,
    function: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a task JSON into ``bridge/queue/`` (does NOT push — caller does)."""
    repo = Path(repo_dir)
    queue = repo / "bridge" / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    spec: Dict[str, Any] = {"task_id": task_id, "kind": kind}
    if kind == "eval":
        spec["code"] = code or ""
    elif kind == "call":
        spec["module"] = module
        spec["function"] = function
        spec["args"] = args or {}
    path = queue / f"{task_id}.json"
    path.write_text(json.dumps(spec, indent=2))
    return path
