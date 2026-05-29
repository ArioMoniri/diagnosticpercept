"""Env detection + smart model loading.

Notebook cells import from this module so fixes here take effect on the
next ``Runtime → Run all`` without requiring a re-import of the .ipynb.
The repo's ``env-check`` cell now does no logic of its own — it just calls
``env_check()`` and ``smart_load_model()`` defined here, after the repo
has been pulled fresh from ``origin/main``.

Auto-detect rules (May 2026):

  GPU class                | quant | max_memory | model
  --------------------------------------------------------
  ≥ 70 GB  (H100/A100-80)  | bf16  | spread if n_gpus>1 | Qwen3.6-27B
  ≥ 48 GB                  | bf16  | spread if n_gpus>1 | Qwen3.6-27B
  ≥ 16 GB  (A100-40, L4)   | NF4   | none (NF4 fits)    | Qwen3.6-27B
  <  16 GB (T4)            | NF4   | none               | Qwen3.5-9B / 4B

Model spread (via ``max_memory``) is only applied when bf16 needs it.
With NF4 (~14 GB) the model fits one GPU; the extras get used by the
H6 parallel path which spawns one worker per GPU.
"""
from __future__ import annotations

import os
import shutil
import sys
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Runtime detection
# ---------------------------------------------------------------------------


def detect_runtime() -> str:
    """Identify whether we're on Colab free / Colab Enterprise / local."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        try:
            import google.colab  # noqa: F401
            return "colab_free"
        except ImportError:
            pass
    if any(k in os.environ for k in ("GOOGLE_CLOUD_PROJECT", "VERTEX_PRODUCT")):
        return "colab_enterprise"
    if "JUPYTERHUB_USER" in os.environ:
        return "jupyterhub"
    return "local"


# ---------------------------------------------------------------------------
# Disk / cache redirect (safe to call multiple times)
# ---------------------------------------------------------------------------


def setup_caches(content_root: str = "/content/.cache") -> Optional[Path]:
    """Point HF / pip / tmp / torch / triton / cuda caches at ``/content``.

    Uses **overwrite** semantics (not ``setdefault``): runtime templates
    sometimes ship with ``HF_HOME=/root/.cache/huggingface`` pre-set, which
    would route the 14 GB model download to the small boot disk and fill it.
    Caught by ml-developer review 2026-05-29 (C1).
    """
    if not Path("/content").exists():
        return None
    root = Path(content_root)
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("pip", "tmp", "huggingface", "transformers", "torch",
                "datasets", "triton", "cuda", "torchinductor"):
        (root / sub).mkdir(exist_ok=True)
    env = {
        "PIP_CACHE_DIR":         str(root / "pip"),
        "TMPDIR":                str(root / "tmp"),
        "HF_HOME":               str(root / "huggingface"),
        "HF_HUB_CACHE":          str(root / "huggingface"),
        "TRANSFORMERS_CACHE":    str(root / "transformers"),
        "TORCH_HOME":            str(root / "torch"),
        "XDG_CACHE_HOME":        str(root),
        # New: datasets cache (load_dataset on MedMCQA writes ~250MB to ~/.cache
        # by default → boot disk). Triton / Inductor / cuBLAS JIT caches grow
        # over a session and would also hit the boot disk.
        "HF_DATASETS_CACHE":     str(root / "datasets"),
        "TRITON_CACHE_DIR":      str(root / "triton"),
        "CUDA_CACHE_PATH":       str(root / "cuda"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "torchinductor"),
    }
    overwrote = []
    for k, v in env.items():
        prev = os.environ.get(k)
        if prev and prev != v:
            overwrote.append(f"{k} (was {prev!r})")
        os.environ[k] = v
    if overwrote:
        print(f"setup_caches: overwrote {len(overwrote)} stale env var(s): "
              f"{', '.join(overwrote[:3])}{'...' if len(overwrote) > 3 else ''}")
    return root


def disk_report(label: str = "") -> None:
    """Print free-space on / and /content (Colab boot vs workspace)."""
    for path in ("/", "/content"):
        if Path(path).exists():
            s = shutil.disk_usage(path)
            free = (s.total - s.used) / 1e9
            warn = "  !! LOW" if free < 10 else ""
            tag = f" [{label}]" if label else ""
            print(f"  disk {path:<10} free={free:6.1f} GB{warn}{tag}")


# ---------------------------------------------------------------------------
# GPU auto-pick
# ---------------------------------------------------------------------------


def auto_pick() -> Dict[str, Any]:
    """Decide model name, quantization, max_memory, N_BENCH based on GPUs."""
    import torch

    # If a previous (old inline) cell set MODEL_OVERRIDE to a Qwen3.5/3.6
    # checkpoint, silently drop it — those don't load on standard
    # transformers (their qwen3_5 architecture is incomplete: vision tower
    # + MTP head + Gated DeltaNet layers all missing handlers). User can
    # still force one via MODEL_OVERRIDE *after* this cell.
    _override = os.environ.get("MODEL_OVERRIDE", "")
    if any(_override.startswith(x) for x in (
        "Qwen/Qwen3.5", "Qwen/Qwen3.6", "Qwen/Qwen3-Next",
    )):
        print(f"Ignoring MODEL_OVERRIDE={_override!r} — that checkpoint "
              f"doesn't load on standard transformers; using auto-pick.")
        del os.environ["MODEL_OVERRIDE"]
        _override = ""

    plan: Dict[str, Any] = {
        "model": _override or "Qwen/Qwen3-14B",  # safe placeholder; auto_pick overrides
        "use_4bit": None,
        "max_memory": None,
        "n_gpus": 0,
        "gpu_gb": 0.0,
        "gpu_name": "",
    }
    if not torch.cuda.is_available():
        plan["use_4bit"] = False
        plan["n_bench"] = 100
        return plan

    plan["n_gpus"] = torch.cuda.device_count()
    plan["gpu_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    plan["gpu_name"] = torch.cuda.get_device_name(0)

    gb = plan["gpu_gb"]
    # User overrides win.
    if "USE_4BIT" in os.environ:
        plan["use_4bit"] = bool(int(os.environ["USE_4BIT"]))
    else:
        # Bumped to 48 GB so A100 40GB (driver reports ~42 GB) uses NF4.
        plan["use_4bit"] = gb < 48.0

    if "MODEL_OVERRIDE" not in os.environ:
        # Qwen3.5/3.6 use a hybrid Gated DeltaNet + Gated Attention arch with
        # multi-token prediction and an integrated vision tower. The checkpoint
        # layer layout (linear_attn.in_proj_qkv, mtp.layers, model.visual.*) is
        # incompatible with what transformers' Qwen3_5ForCausalLM tries to
        # load — most weights end up MISSING/UNEXPECTED. Until transformers
        # ships a Qwen3.5/3.6-specific causal class, default to Qwen3 which
        # loads cleanly on standard transformers everywhere.
        if gb >= 70:
            plan["model"] = "Qwen/Qwen3-32B"   # bf16 ~64 GB, fits one A100-80
        elif gb >= 24:
            plan["model"] = "Qwen/Qwen3-14B"   # bf16 ~28 GB, fits A100-40
        elif gb >= 12:
            plan["model"] = "Qwen/Qwen3-8B"
        else:
            plan["model"] = "Qwen/Qwen3-4B"

    # max_memory ONLY when bf16 needs spreading (multi-GPU, large model).
    # When NF4, the model is ~14 GB — fits one GPU; spreading just adds
    # cross-GPU collective overhead.
    if not plan["use_4bit"] and plan["n_gpus"] > 1:
        # 25% per GPU works for 80 GB cards (4×20=80 GB total cap, holds
        # 64 GB bf16 27B + headroom). Floor at 20 GiB so we don't undershoot.
        per_gpu = max(20, int(gb * 0.25))
        plan["max_memory"] = {i: f"{per_gpu}GiB" for i in range(plan["n_gpus"])}

    # N_BENCH default by GPU class.
    if "N_BENCH" in os.environ:
        plan["n_bench"] = int(os.environ["N_BENCH"])
    elif gb >= 70:
        plan["n_bench"] = 1273
    elif gb >= 36:
        plan["n_bench"] = 1273           # NF4 27B is fast enough on A100-40
    elif gb >= 12:
        plan["n_bench"] = 600
    else:
        plan["n_bench"] = 300

    return plan


def env_check() -> Dict[str, Any]:
    """One-shot env check that returns the plan + prints a banner."""
    import torch

    runtime = detect_runtime()
    setup_caches()

    print(f"Runtime: {runtime}")
    print(f"Python : {sys.version.split()[0]}  Torch: {torch.__version__}")
    print(f"CUDA   : {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    disk_report("start")

    plan = auto_pick()
    print(f"GPUs   : {plan['n_gpus']}× {plan['gpu_name']}  ({plan['gpu_gb']:.1f} GB each)")
    print(f"Plan   : model={plan['model']}  use_4bit={plan['use_4bit']}  "
          f"max_memory={plan['max_memory']}  N_BENCH={plan['n_bench']}")
    return plan


# ---------------------------------------------------------------------------
# Smart model loader
# ---------------------------------------------------------------------------


def smart_load_model(plan: Optional[Dict[str, Any]] = None) -> Tuple[Any, str]:
    """Load the model the plan picked, with CUDA-cache hygiene and retries.

    Returns ``(lm, model_name)``. On a "CUDA device busy / unavailable" error
    (typically leftover state from a prior failed allocation), retries once
    after clearing the CUDA cache.
    """
    import torch
    from .model import load_first_available

    if plan is None:
        plan = auto_pick()

    candidates = [plan["model"]]
    token = os.environ.get("HF_TOKEN")

    def _attempt(quantize: Optional[bool] = None, force_single_gpu: bool = False):
        kw = dict(
            candidates=candidates,
            token=token,
            quantize_4bit=plan["use_4bit"] if quantize is None else quantize,
            max_memory=plan.get("max_memory"),
        )
        if force_single_gpu:
            kw["max_memory"] = None
            kw["device_map"] = "cuda:0"
        return load_first_available(**kw)

    def _full_cuda_reset():
        if not torch.cuda.is_available():
            return
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        # Reset peak stats — purely cosmetic but signals a clean slate.
        try:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        except Exception:
            pass

    def _is_recoverable(e: Exception) -> bool:
        msg = str(e).lower()
        return any(s in msg for s in (
            "busy or unavailable", "cublas_status_alloc_failed",
            "cuda error", "automatic conversion of the weights",
            "dispatched on the cpu or the disk",
        ))

    # Ladder of attempts. Ordering rationale:
    #   1. primary (the plan)
    #   2. retry after CUDA reset (most failures are stale-context)
    #   3. NF4 single-GPU — safest VRAM footprint; bf16-single skipped
    #      because for a 14B model on 40GB it strictly OOMs on backward.
    # The previous "bf16-single-GPU" attempt was a redundant subset of
    # NF4-single and never succeeded where the others failed (C4).
    lm = name = None
    last_exc: Optional[Exception] = None
    for attempt_label, kw in [
        ("primary",                dict()),
        ("retry after CUDA reset", dict()),  # same args, after reset
        ("NF4 single-GPU",         dict(quantize=True, force_single_gpu=True)),
    ]:
        try:
            print(f"\n[smart_load] attempt: {attempt_label}")
            if attempt_label == "retry after CUDA reset":
                _full_cuda_reset()
            lm, name = _attempt(**kw)
            break
        except (RuntimeError, ValueError) as e:
            last_exc = e
            print(f"  → failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
            if not _is_recoverable(e):
                raise
    if lm is None:
        # Last resort: drop one model size and retry. Louder warning (C5).
        smaller = {"Qwen/Qwen3-32B": "Qwen/Qwen3-14B",
                    "Qwen/Qwen3-14B": "Qwen/Qwen3-8B",
                    "Qwen/Qwen3-8B":  "Qwen/Qwen3-4B"}.get(plan["model"])
        if smaller:
            print()
            print("!" * 72)
            print(f"! DOWNGRADE: all 3 attempts on {plan['model']} failed.")
            print(f"! Falling back to {smaller} (smaller model, NF4 quant).")
            print(f"! Set MODEL_OVERRIDE='{plan['model']}' to retry without downgrade.")
            print("!" * 72)
            _full_cuda_reset()
            candidates = [smaller]
            lm, name = _attempt(quantize=True, force_single_gpu=True)
        else:
            raise last_exc or RuntimeError("smart_load_model: no candidates succeeded")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1e9
        print(f"\n>>> Loaded: {name}")
        print(f"    layers = {lm.n_layers}")
        print(f"    d_ff   = {lm.d_ff}")
        print(f"    dtype  = {lm.dtype}")
        print(f"    device = {lm.device}")
        print(f"    VRAM (GPU 0): {used:.2f} GB allocated")
    return lm, name


# ---------------------------------------------------------------------------
# Repo self-update — call at the top so subsequent `from src...` get latest.
# ---------------------------------------------------------------------------


def hard_reset_repo(repo_dir: str) -> str:
    """``git fetch + reset --hard origin/main`` then drop cached src.* imports."""
    import subprocess
    subprocess.run(["git", "-C", repo_dir, "fetch", "origin", "main"], check=False)
    subprocess.run(["git", "-C", repo_dir, "reset", "--hard", "origin/main"], check=False)
    sha = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    # Drop cached src.* modules so the next import pulls the new code.
    for m in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
        del sys.modules[m]
    importlib.invalidate_caches()
    print(f"Repo @ {sha}")
    return sha
