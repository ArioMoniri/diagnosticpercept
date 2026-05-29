#!/usr/bin/env python3
"""Push a bridge task JSON to ``bridge/queue/`` and `git push` it.

The git-bus bridge (``src.bridge``) watches ``bridge/queue/*.json`` and runs
each task on the Colab side. Writing those JSONs by hand is verbose and
error-prone; this CLI handles the boilerplate.

Examples
--------

Run a Python expression on Colab and dump a variable to ``results/``::

    python scripts/push_task.py eval \\
        --code 'len(lm.layers)' \\
        --save 'n_layers=results/n_layers.json'

Call a module function with args::

    python scripts/push_task.py call \\
        --module src.sycophancy --function run_sycophancy_probe \\
        --args '{"n_questions": 50, "seed": 0}' \\
        --needs lm,items \\
        --save 'cases=results/sycophancy_cases.json'

Re-run an existing spec (e.g. retry a failed log)::

    python scripts/push_task.py replay bridge/log/20260529-114500-h6.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "bridge" / "queue"


def _ts_slug(label: str) -> str:
    """``20260529-114503-label`` — lexicographically sortable task IDs."""
    return time.strftime("%Y%m%d-%H%M%S-", time.gmtime()) + label


def _parse_save(items: Optional[List[str]]) -> Dict[str, str]:
    """Parse ``--save key=path`` options into ``{key: path}``."""
    out: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--save expects key=path, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_needs(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def _git_commit_push(task_path: Path, label: str) -> None:
    """Stage the new task JSON, commit, and push to ``origin/main``."""
    subprocess.run(["git", "-C", str(ROOT), "add", str(task_path)], check=True)
    msg = f"bridge: enqueue {label}"
    subprocess.run(["git", "-C", str(ROOT), "commit", "-m", msg], check=True)
    subprocess.run(["git", "-C", str(ROOT), "push", "origin", "HEAD"], check=True)


def _write_task(spec: Dict[str, Any], label: str, dry_run: bool) -> Path:
    QUEUE.mkdir(parents=True, exist_ok=True)
    task_id = _ts_slug(label)
    spec.setdefault("task_id", task_id)
    out = QUEUE / f"{task_id}.json"
    out.write_text(json.dumps(spec, indent=2))
    print(f"[push_task] wrote {out}")
    if dry_run:
        print("[push_task] --dry-run set; not committing or pushing.")
    else:
        _git_commit_push(out, label)
    return out


def cmd_eval(args: argparse.Namespace) -> None:
    spec = {
        "kind": "eval",
        "code": args.code,
        "needs": _parse_needs(args.needs),
        "save": _parse_save(args.save),
    }
    _write_task(spec, label=args.label or "eval", dry_run=args.dry_run)


def cmd_call(args: argparse.Namespace) -> None:
    fn_args: Dict[str, Any] = json.loads(args.args) if args.args else {}
    spec = {
        "kind": "call",
        "module": args.module,
        "function": args.function,
        "args": fn_args,
        "needs": _parse_needs(args.needs),
        "save": _parse_save(args.save),
    }
    label = args.label or f"{args.module.split('.')[-1]}-{args.function}"
    _write_task(spec, label=label, dry_run=args.dry_run)


def cmd_replay(args: argparse.Namespace) -> None:
    src = Path(args.path)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    spec = json.loads(src.read_text())
    # If we're replaying a log, strip log-only fields.
    for k in ("stdout", "stderr", "result", "traceback", "task_id"):
        spec.pop(k, None)
    label = args.label or src.stem.split("-", 2)[-1]
    _write_task(spec, label=label, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Push a bridge task to bridge/queue/.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write the JSON but don't commit or push.")
    p.add_argument("--label", help="Slug appended to the task id (for grep).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="kind=eval — run an arbitrary code string.")
    pe.add_argument("--code", required=True, help="Python source to eval/exec.")
    pe.add_argument("--needs", help="Comma-separated globals to inject (e.g. 'lm,items').")
    pe.add_argument("--save", action="append",
                    help="key=path; dump a global to disk after the task. Repeatable.")
    pe.set_defaults(func=cmd_eval)

    pc = sub.add_parser("call", help="kind=call — invoke module.function(**args).")
    pc.add_argument("--module", required=True)
    pc.add_argument("--function", required=True)
    pc.add_argument("--args", help="JSON dict of kwargs.")
    pc.add_argument("--needs", help="Comma-separated globals to inject.")
    pc.add_argument("--save", action="append",
                    help="key=path; dump a global to disk after the task. Repeatable.")
    pc.set_defaults(func=cmd_call)

    pr = sub.add_parser("replay", help="Re-enqueue an existing task/log JSON.")
    pr.add_argument("path", help="Path to a bridge/queue/*.json or bridge/log/*.json.")
    pr.set_defaults(func=cmd_replay)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
