"""Generate notebooks/diagnostic_percept.ipynb.

The notebook is the user-facing Colab entry point. It clones this repo at the
top so a fresh Colab can run it standalone, imports from ``src/``, and persists
all artifacts under ``/content/results/``.

Each section is wrapped in try/except + traceback so Colab failures surface
the full stack inline (Colab cell-output truncation can otherwise hide them).

Run with::

    python scripts/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "diagnostic_percept.ipynb"
OUT.parent.mkdir(exist_ok=True)


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s)


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip("\n"))


def wrap(label: str, body: str) -> str:
    """Wrap a code body in try/except with full traceback printing."""
    indented = "\n".join("    " + line if line else "" for line in body.splitlines())
    return (
        f"# === {label} ===\n"
        "import traceback\n"
        "try:\n"
        f"{indented}\n"
        "except Exception:\n"
        "    print('!!!!!!!!!! CELL FAILED — full traceback below !!!!!!!!!!')\n"
        "    traceback.print_exc()\n"
        "    raise\n"
    )


REPO_URL = "https://github.com/ArioMoniri/diagnosticpercept.git"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "machine_shape": "hm"},
    "accelerator": "GPU",
}

cells = []

cells.append(md(f"""# Diagnostic Percept — single-neuron diagnosis intervention

Port of Kazemi et al. 2026 (\"A Single Neuron Is Sufficient to Bypass Safety
Alignment in LLMs\", [arXiv 2605.08513](https://arxiv.org/abs/2605.08513)) from
safety to clinical diagnosis. Three hypotheses on a medical LLM:

- **H1** — a single MLP neuron whose suppression flips committed diagnosis to hedging.
- **H2** — disease-specific concept neurons whose amplification injects disease content into benign prompts.
- **H3** — symptom→diagnosis routing visible via residual-stream activation patching.

This notebook clones [{REPO_URL}]({REPO_URL}), imports from `src/`, and writes
all outputs to `/content/results/`. Every section is wrapped in try/except with
full traceback printing so Colab failures are visible.
"""))

cells.append(md("## 1. Setup — install, GPU check, clone, HF login"))

cells.append(code(
    "# Boot disk on Vertex AI Colab Enterprise is ~101 GB and starts ~60 GB\n"
    "# full (system image). /content is the 527 GB workspace. Without\n"
    "# redirection, pip's temp build files + pip cache + HF cache all land\n"
    "# on the boot disk and can fill it during the install — at which point\n"
    "# Vertex AI health checks fail and the runtime is marked unhealthy.\n"
    "# Set EVERY cache dir to /content BEFORE the first pip call.\n"
    "import os, subprocess, sys, shutil\n"
    "from pathlib import Path\n"
    "_C = Path('/content/.cache') if Path('/content').exists() else None\n"
    "if _C:\n"
    "    _C.mkdir(parents=True, exist_ok=True)\n"
    "    (_C / 'pip').mkdir(exist_ok=True)\n"
    "    (_C / 'tmp').mkdir(exist_ok=True)\n"
    "    os.environ['PIP_CACHE_DIR']  = str(_C / 'pip')\n"
    "    os.environ['TMPDIR']         = str(_C / 'tmp')\n"
    "    os.environ['HF_HOME']        = str(_C / 'huggingface')\n"
    "    os.environ['HF_HUB_CACHE']   = str(_C / 'huggingface')\n"
    "    os.environ['TRANSFORMERS_CACHE'] = str(_C / 'transformers')\n"
    "    os.environ['TORCH_HOME']     = str(_C / 'torch')\n"
    "    os.environ['XDG_CACHE_HOME'] = str(_C)\n"
    "    print(f'Caches → {_C} (boot disk is small; this is mandatory)')\n"
    "\n"
    "def _disk(label=''):\n"
    "    for p in ('/', '/content'):\n"
    "        if Path(p).exists():\n"
    "            s = shutil.disk_usage(p)\n"
    "            free = (s.total - s.used) / 1e9\n"
    "            print(f'  [{label}] disk {p:<10} free={free:6.1f} GB')\n"
    "_disk('start')\n"
    "\n"
    "# Surgical upgrade: install transformers main with --no-deps so it does\n"
    "# NOT pull a newer torch / torchvision / pillow. Then pin transformers'\n"
    "# runtime deps to the *exact* versions it expects (it pins tokenizers\n"
    "# <=0.23.0, which a bare `--upgrade tokenizers` overshoots to 0.23.1).\n"
    "def _pip(*args):\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], check=False)\n"
    "\n"
    "# 1. transformers main, no cascading dep upgrades\n"
    "_pip('--upgrade', '--no-deps',\n"
    "     'transformers @ git+https://github.com/huggingface/transformers.git@main')\n"
    "# 2. transformers' runtime deps. huggingface_hub MUST come from main\n"
    "#    too — transformers main imports `is_offline_mode` which only\n"
    "#    exists in hub's main branch (older released versions removed it,\n"
    "#    newer renamed it). Install hub from git@main to match.\n"
    "_pip('--no-deps', '--upgrade',\n"
    "     'huggingface_hub @ git+https://github.com/huggingface/huggingface_hub.git@main',\n"
    "     'safetensors>=0.4',\n"
    "     'tokenizers>=0.22.0,<=0.23.0',\n"
    "     'regex',\n"
    "     'requests',\n"
    "     'pyyaml',\n"
    "     'httpx',\n"
    "     'filelock')\n"
    "# 3. our other libs --no-deps (accelerate / bitsandbytes happy w/ Colab torch)\n"
    "_pip('--upgrade', '--no-deps', 'accelerate>=0.34', 'bitsandbytes>=0.43')\n"
    "# 4. plain installs of small libs (no risk to torch/pillow)\n"
    "_pip('scikit-learn', 'matplotlib', 'tqdm', 'datasets', 'nbformat', 'ipywidgets')\n"
    "_disk('after step 4')\n"
    "# 5. Pillow self-heal if a prior run pulled pillow 12 (PIL.ImageText breaks).\n"
    "try:\n"
    "    import PIL.ImageText  # canary for pillow 12 ABI break\n"
    "except Exception:\n"
    "    print('Repairing pillow (pinning <12) ...')\n"
    "    _pip('--force-reinstall', '--no-deps', 'pillow<12')\n"
    "\n"
    "# Free pip cache to reclaim disk now that everything is installed.\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'cache', 'purge'], check=False, capture_output=True)\n"
    "_disk('after purge')\n"
    "\n"
    "# Drop the pre-imported transformers + huggingface_hub from Colab so the\n"
    "# re-import picks up the new versions.\n"
    "import importlib\n"
    "for m in [k for k in list(sys.modules)\n"
    "          if k in ('transformers', 'huggingface_hub')\n"
    "          or k.startswith('transformers.') or k.startswith('huggingface_hub.')]:\n"
    "    del sys.modules[m]\n"
    "importlib.invalidate_caches()\n"
    "\n"
    "# Self-heal: if the import still fails because hub<->transformers got\n"
    "# out of sync, re-install both from main and retry once.\n"
    "try:\n"
    "    import transformers\n"
    "except ImportError as _e:\n"
    "    print(f'Self-healing transformers/hub mismatch: {_e}')\n"
    "    _pip('--no-deps', '--upgrade', '--force-reinstall',\n"
    "         'transformers @ git+https://github.com/huggingface/transformers.git@main',\n"
    "         'huggingface_hub @ git+https://github.com/huggingface/huggingface_hub.git@main')\n"
    "    for m in [k for k in list(sys.modules)\n"
    "              if k in ('transformers', 'huggingface_hub')\n"
    "              or k.startswith('transformers.') or k.startswith('huggingface_hub.')]:\n"
    "        del sys.modules[m]\n"
    "    importlib.invalidate_caches()\n"
    "    import transformers\n"
    "_has_q35 = hasattr(transformers, 'Qwen3_5ForCausalLM')\n"
    "print(f'transformers {transformers.__version__}  Qwen3_5 registered: {_has_q35}')\n"
    "if not _has_q35:\n"
    "    # NB: do NOT auto-restart the kernel here. Vertex AI's idle detector\n"
    "    # interprets the post-restart wait as inactivity and may shut the VM\n"
    "    # down within minutes. Instead, halt cleanly with a clear message so\n"
    "    # the user does the restart manually and immediately Run All again.\n"
    "    raise SystemExit(\n"
    "        '\\n' + '=' * 70 +\n"
    "        '\\n  ACTION REQUIRED: restart the kernel, then click Run All again.'\n"
    "        '\\n  Colab Enterprise: Runtime → Restart session → Run all.'\n"
    "        '\\n  (Auto-restart removed because Vertex AI counts the post-'\n"
    "        '\\n   restart idle time toward the auto-shutdown timer.)'\n"
    "        '\\n' + '=' * 70\n"
    "    )"
))

# EMERGENCY DISK RECOVERY — run only if /  (boot disk) is filling up.
# Wipes pip cache + HF cache + tmp on the boot disk and re-points to /content.
cells.append(code("""
# Optional: run this cell ONLY if the boot disk (/) is near full and the
# normal cache redirect happened too late. Safe to leave commented out.
# import subprocess, shutil, os
# subprocess.run(['rm', '-rf', '/root/.cache/pip'], check=False)
# subprocess.run(['rm', '-rf', '/root/.cache/huggingface'], check=False)
# subprocess.run(['rm', '-rf', '/root/.cache/torch'], check=False)
# subprocess.run(['rm', '-rf', '/tmp/pip*'], check=False)
# print('Boot-disk caches wiped.')
"""))

# Set CUDA alloc config BEFORE torch imports anywhere — must be very first.
cells.append(code("""
import os
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Redirect HF / Torch caches to /content (Colab Enterprise's 195 GB workspace
# disk) so model weights don't fill the ~90 GB boot disk. Must happen before
# transformers / huggingface_hub are imported, so set it here.
_CACHE_ROOT = '/content/.cache' if Path('/content').exists() else None
if _CACHE_ROOT:
    Path(_CACHE_ROOT).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME',          f'{_CACHE_ROOT}/huggingface')
    os.environ.setdefault('TRANSFORMERS_CACHE', f'{_CACHE_ROOT}/transformers')
    os.environ.setdefault('TORCH_HOME',       f'{_CACHE_ROOT}/torch')
    os.environ.setdefault('XDG_CACHE_HOME',   _CACHE_ROOT)
    print(f'Caches redirected to {_CACHE_ROOT}')
else:
    print('No /content workspace (not on Colab); using default cache dirs.')

# Force tqdm.notebook so progress bars render as Colab widgets, not raw lines
# (matters for the long H6/H7/sycophancy passes).
try:
    import tqdm, tqdm.notebook
    tqdm.tqdm = tqdm.notebook.tqdm
    import tqdm.auto
    tqdm.auto.tqdm = tqdm.notebook.tqdm
    print('tqdm.notebook installed as the default tqdm')
except Exception as _e:
    print('tqdm.notebook unavailable, keeping default:', _e)
"""))

cells.append(code(wrap("env check", f"""
import os, sys, subprocess, json, time, traceback, importlib
from pathlib import Path
import torch

# Runtime detection — free Colab vs Colab Enterprise (Vertex Workbench) vs other.
def _detect_runtime():
    if 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_GPU' in os.environ:
        try:
            import google.colab  # noqa: F401
            return 'colab_free'
        except ImportError:
            pass
    if any(k in os.environ for k in ('GOOGLE_CLOUD_PROJECT', 'VERTEX_PRODUCT')):
        return 'colab_enterprise'
    if 'JUPYTERHUB_USER' in os.environ:
        return 'jupyterhub'
    return 'local'
RUNTIME = _detect_runtime()
print(f'Runtime: {{RUNTIME}}')
print('Python:', sys.version.split()[0])
print('Torch :', torch.__version__)
print('CUDA  :', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')
if torch.cuda.is_available():
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f'GPU: {{gpu_name}}  | Memory: {{gpu_gb:.1f}} GB')

    # NOTE: do NOT set MODEL_OVERRIDE / USE_4BIT / N_BENCH here. All
    # decisioning lives in src/setup.py auto_pick(), which is called by
    # smart_load_model() in the model-load cell. If you set env vars
    # here, an old snapshot of THIS cell (frozen in your imported .ipynb)
    # could write a stale Qwen3.5/3.6 pick that auto_pick can't override
    # because the env var "wins". src/setup.py also actively strips
    # MODEL_OVERRIDE if it points to a known-broken Qwen3.5/3.6 checkpoint.

# Disk sanity. Colab Enterprise's boot disk is ~94 GB and starts ~90% full
# (system image). /content is the 195 GB workspace where caches go.
import shutil
for path in ('/', '/content'):
    if Path(path).exists():
        s = shutil.disk_usage(path)
        used_pct = 100 * s.used / s.total
        warn = ' !! LOW' if (s.total - s.used) < 10 * (1024**3) else ''
        print(f'Disk {{path:<10}}  {{s.used/1e9:6.1f}} / {{s.total/1e9:6.1f}} GB  ({{used_pct:.0f}}%){{warn}}')

# Validate cache redirect — the model download (~14 GB at NF4, ~54 GB at bf16)
# MUST land on /content or the boot disk fills up.
_hf_home = os.environ.get('HF_HOME', '')
if _hf_home and not _hf_home.startswith('/content'):
    print('!! WARN: HF_HOME is', _hf_home, '— model will download to boot disk!')
elif _hf_home:
    print(f'HF cache → {{_hf_home}}  (/content has plenty of room)')
else:
    print('!! WARN: HF_HOME not set; model download will use ~/.cache (boot disk).')

REPO_URL = '{REPO_URL}'
REPO_DIR = 'diagnosticpercept'
if not Path(REPO_DIR).exists():
    print('Cloning', REPO_URL, '...')
    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)
else:
    # Hard-reset to origin/main so re-runs always pick up the latest code.
    print('Fetching + hard-resetting to origin/main ...')
    subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin', 'main'], check=False)
    subprocess.run(['git', '-C', REPO_DIR, 'reset', '--hard', 'origin/main'], check=False)

# Print current SHA so we can verify the running version.
sha = subprocess.run(['git', '-C', REPO_DIR, 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
print(f'Repo @ commit: {{sha}}  (expect 8635794 or newer for H4+H5)')

# Drop any previously-imported src.* modules so Python re-loads from disk —
# a kernel re-run with the prior clone may have cached the old discover.py.
for m in [k for k in list(sys.modules) if k == 'src' or k.startswith('src.')]:
    del sys.modules[m]
importlib.invalidate_caches()

repo_path = str(Path(REPO_DIR).resolve())
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

RESULTS = Path('/content/results'); RESULTS.mkdir(parents=True, exist_ok=True)
print('Repo   :', repo_path)
print('Results:', RESULTS)
""")))

cells.append(code(wrap("preflight — print the run plan", """
# Single-glance summary of what's about to happen so you can abort before
# downloading 14 GB of model weights if anything is wrong.
print('=' * 62)
print(f'  Runtime       : {RUNTIME}')
print(f'  GPU           : {"none" if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}  ({"-" if not torch.cuda.is_available() else f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB"})')
print(f'  Model         : {os.environ.get("MODEL_OVERRIDE", "(auto-pick from chain)")}')
print(f'  Quantize 4bit : {os.environ.get("USE_4BIT", "auto")}')
print(f'  N_BENCH       : {os.environ.get("N_BENCH", "default")}')
print(f'  HF cache      : {os.environ.get("HF_HOME", "(default ~/.cache)")}')
print()
print('  Estimated wall time on this hardware:')
_n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
_gpu_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if _n_gpu else 0
_gpu_name = torch.cuda.get_device_name(0) if _n_gpu else ''
# H100 ≈ 1.5× A100 fwd throughput. Wall time scales by 1/n_gpu for H6.
_is_h100 = 'H100' in _gpu_name
_throughput_factor = 1.0 if _is_h100 else 1.5  # A100 vs H100
_par = max(1, _n_gpu)
print(f'  Hardware: {_n_gpu}× {_gpu_name or "CPU"}  (parallel factor {_par})')
if _gpu_gb >= 36:
    h1   = round(5  * _throughput_factor, 1)            # H1 stays single-GPU
    h6   = round(75 * _throughput_factor / _par, 1)      # parallelized
    h7   = round(3  * _throughput_factor, 1)             # single-GPU
    syc  = round(15 * _throughput_factor, 1)             # single-GPU for now
    total = h1 + h6 + h7 + syc
    print(f'    H1 discover           ~{h1} min  (single GPU)')
    print(f'    H6 deep (1273×3)      ~{h6} min  ({_par}× parallel)')
    print(f'    H7 (300 items)        ~{h7} min  (single GPU)')
    print(f'    H8 + sycophancy       ~{syc} min  (single GPU)')
    print(f'    --- TOTAL             ~{total/60:.1f} hr')
else:
    print('    Small-GPU budget — auto-pick will drop model size.')
print('=' * 62)
""")))

cells.append(code(wrap("HF login (optional — only for gated models)", """
import os
# Qwen3 is open-weights and needs NO token. HF_TOKEN is only needed if you
# override to a gated model. Resolution order:
#   1. env var HF_TOKEN
#   2. Colab secret HF_TOKEN
#   3. interactive notebook_login() widget (may not render in all Colab
#      runtimes — if so, use the manual paste cell that follows)
def _resolve_hf_token():
    if os.environ.get('HF_TOKEN'):
        print('HF_TOKEN already set in env.')
        return
    # Free Colab has google.colab.userdata; Colab Enterprise does NOT.
    if RUNTIME == 'colab_free':
        try:
            from google.colab import userdata
            tok = userdata.get('HF_TOKEN')
            if tok:
                os.environ['HF_TOKEN'] = tok
                print('HF_TOKEN loaded from Colab secret.')
                return
        except Exception:
            pass
    print('No HF_TOKEN in env.')
    if RUNTIME == 'colab_enterprise':
        print('Colab Enterprise: set HF_TOKEN as a runtime-template env var,')
        print('or paste into the manual cell below.')
    else:
        print('Qwen3 is open-weights so this is fine to skip for the default chain.')
    try:
        from huggingface_hub import notebook_login
        notebook_login()
        print('Token widget rendered above ↑ (paste + Login).')
        print('If you do not see a widget, use the manual paste cell below.')
    except Exception as e:
        print(f'(notebook_login unavailable: {e})')

_resolve_hf_token()
""")))

# Manual paste fallback — separate cell so widget failures don't block.
cells.append(code(wrap("HF token: manual paste fallback (skip if widget worked)", """
# If the widget above didn't render, paste your token below between the quotes
# and run THIS cell. Leave blank to skip.
HF_TOKEN_PASTE = ''   # ← paste like 'hf_xxxxxxxxxxxxxxxxx', then Run cell

if HF_TOKEN_PASTE.strip():
    os.environ['HF_TOKEN'] = HF_TOKEN_PASTE.strip()
    print(f'HF_TOKEN set manually ({len(HF_TOKEN_PASTE.strip())} chars).')
else:
    print('No manual token pasted. Continuing with whatever the previous cell resolved.')
""")))

cells.append(md(
    "## 2. Load model + verify hooks\n\n"
    "Default chain is **Qwen-only**: tries the newest Qwen3.5 variants, then "
    "Qwen3-32B → 14B → 8B → 4B. The first that fits VRAM (with NF4 4-bit "
    "below 24 GB) wins. Patches every MLP forward to expose "
    "`h = SiLU(W_gate x) * (W_up x)` with `retain_grad`.\n\n"
    "*To force a specific Qwen variant*: set "
    "`os.environ['MODEL_OVERRIDE'] = 'Qwen/<exact-repo-name>'` "
    "**before** running this cell."
))

cells.append(code(wrap("load model", """
# All decision logic lives in src/setup.py — fixes to GPU detection, model
# auto-pick, or max_memory take effect on the next Run All without
# re-importing the notebook (the env-check cell pulls latest src/ first).
from src.setup import smart_load_model
from src.model import set_seed
set_seed(0)

lm, MODEL_NAME = smart_load_model()
# Legacy globals so downstream cells keep working.
USE_4BIT = bool(int(os.environ.get('USE_4BIT', '0')))
N_BENCH  = int(os.environ.get('N_BENCH', '1273'))
n_gpus   = torch.cuda.device_count() if torch.cuda.is_available() else 0
""")))

# (Legacy inline load block removed — smart_load_model handles everything.)

cells.append(code(wrap("sanity: h.retain_grad flows", """
if torch.cuda.is_available():
    torch.cuda.empty_cache()
ids = lm.tokenizer('Chest pain. Diagnosis:', return_tensors='pt').input_ids.to(lm.device)
lm.model.zero_grad(set_to_none=True)
with torch.enable_grad():
    out = lm.model(input_ids=ids, use_cache=False)
    # logit at last position only — no need for full vocab sum.
    out.logits[0, -1, 0].backward()
g = lm.layers[0].mlp._h.grad
assert g is not None, 'h.grad is None — hook patching failed.'
assert torch.isfinite(g).all(), 'h.grad has non-finite values.'
assert g.abs().sum() > 0, 'h.grad is all zeros.'
print('OK: layer-0 h.grad shape', tuple(g.shape), 'nonzero =', (g.abs() > 0).sum().item())
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'  VRAM after sanity: {torch.cuda.memory_allocated()/1e9:.2f} GB')
""")))

cells.append(md(
    "## 3. H1 — diagnosis-gate neuron\n\n"
    "Contrastive set (pathognomonic vs ambiguous vignettes), gradient × activation "
    "discovery (Eqs. 2-4), multiplier sweep over top-5 candidates, capability "
    "check on a tiny MedQA set, activation-distribution plot."
))

cells.append(code(wrap("H1 — load data, discover", """
from src.data import build_h1
from src.discover import discover, sweep, best_multiplier, NeuronScore, DEFAULT_M_SWEEP

H1_RESULTS = RESULTS / 'h1'; H1_RESULTS.mkdir(exist_ok=True)
h1 = build_h1()
print(f'Positive vignettes: {len(h1["positive"])}')
print(f'Negative vignettes: {len(h1["negative"])}')

cands_path = H1_RESULTS / 'candidates.json'
if cands_path.exists():
    cands_raw = json.loads(cands_path.read_text())
    cands = [NeuronScore(**c) for c in cands_raw]
    print(f'Reloaded {len(cands)} candidates from {cands_path}')
else:
    t0 = time.time()
    cands = discover(
        lm, positive_prompts=h1['positive'], negative_prompts=h1['negative'],
        target_phrases=h1['commitment_phrases'], icd10_tokens=h1['icd10_tokens'],
        layer_range=None, top_k=5,
    )
    print(f'Discovery done in {time.time()-t0:.1f}s')
    cands_path.write_text(json.dumps([c.__dict__ for c in cands], indent=2))

print('\\nTop-5 candidates:')
for c in cands:
    print(f'  L{c.layer:>2}:F{c.neuron:<6}  score={c.score:+.4f}  a_pos={c.a_pos:+.3f}  a_neg={c.a_neg:+.3f}')
""")))

cells.append(code(wrap("H1 — multiplier sweep over top-5 + capability cost", """
from types import SimpleNamespace
from src.discover import best_multiplier_with_capability, mean_target_logprob_under, _target_first_token_ids, _mean_target_logprob

sweep_path = H1_RESULTS / 'sweep.json'
cap_path = H1_RESULTS / 'capability_sweep.json'
probe = h1['positive'][:8]
hard_for_capability = h1['hard_cases'][:8]  # cases the unmodified model handles

if sweep_path.exists():
    sweep_raw = json.loads(sweep_path.read_text())
    print(f'Reloaded sweep ({len(sweep_raw)} rows)')
else:
    sw = sweep(
        lm, candidates=cands, probes=probe,
        target_phrases=h1['commitment_phrases'], icd10_tokens=h1['icd10_tokens'],
        multipliers=DEFAULT_M_SWEEP, sample_prompt=h1['positive'][0],
    )
    sweep_raw = [s.__dict__ for s in sw]
    sweep_path.write_text(json.dumps(sweep_raw, indent=2))

# Capability sweep: same (cand × m) grid, but evaluated on the HARD cases.
# Drops in this log-prob = capability loss under the intervention.
if cap_path.exists():
    capability_lp = {tuple(eval(k)): v for k, v in json.loads(cap_path.read_text()).items()}
    print(f'Reloaded capability sweep ({len(capability_lp)} entries)')
else:
    capability_lp = mean_target_logprob_under(
        lm, candidates=cands, capability_prompts=hard_for_capability,
        target_phrases=h1['commitment_phrases'], icd10_tokens=h1['icd10_tokens'],
        multipliers=DEFAULT_M_SWEEP,
    )
    cap_path.write_text(json.dumps({str(k): v for k, v in capability_lp.items()}, indent=2))

# Baseline (no intervention) capability log-prob.
target_ids = _target_first_token_ids(lm.tokenizer, h1['commitment_phrases'], h1['icd10_tokens'])
baseline_cap = _mean_target_logprob(lm, hard_for_capability, target_ids)
print(f'Baseline capability log-prob (no intervention): {baseline_cap:+.4f}')

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
by_n_sup = {}
by_n_cap = {}
for s in sweep_raw:
    k = (s['layer'], s['neuron'])
    by_n_sup.setdefault(k, []).append((s['multiplier'], s['target_logprob']))
for (L, N, m), lp in capability_lp.items():
    by_n_cap.setdefault((L, N), []).append((m, lp))
for k, pts in by_n_sup.items():
    pts.sort(); xs, ys = zip(*pts); axes[0].plot(xs, ys, marker='o', label=f'L{k[0]}:F{k[1]}')
for k, pts in by_n_cap.items():
    pts.sort(); xs, ys = zip(*pts); axes[1].plot(xs, ys, marker='o', label=f'L{k[0]}:F{k[1]}')
axes[0].set_title('suppression on probe set (lower = more suppressed)')
axes[1].set_title('capability on hard cases (lower = more capability lost)')
for ax in axes:
    ax.set_xlabel('multiplier m'); ax.set_ylabel('mean target log-prob')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(H1_RESULTS / 'sweep.png', dpi=140); plt.show()

# Composite m* selection: suppression - lambda * capability_cost.
sweep_objs = [SimpleNamespace(**s) for s in sweep_raw]
L_star, N_star, m_star, composite = best_multiplier_with_capability(
    sweep_objs, capability_lp, baseline_cap, lambda_cap=1.0,
)
print(f'\\nBest gate (capability-aware): L{L_star}:F{N_star} at m={m_star}')
naive_L, naive_N, naive_m = best_multiplier(sweep_objs)
print(f'  (naive max-suppression would have picked: L{naive_L}:F{naive_N} at m={naive_m})')
""")))

cells.append(code(wrap("H1 — activation distribution at winning neuron", """
import numpy as np
from src.model import clear_h

acts_pos, acts_neg = [], []
for prompts, bucket in [(h1['positive'], acts_pos), (h1['negative'], acts_neg)]:
    for p in prompts:
        enc = lm.tokenizer(p, return_tensors='pt').to(lm.device)
        with torch.no_grad():
            lm.model(input_ids=enc.input_ids, use_cache=False)
        h = lm.layers[L_star].mlp._h[0, :, N_star].detach().float().cpu().numpy()
        bucket.extend(h.tolist())
        clear_h(lm.layers)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(acts_pos, bins=40, alpha=0.6, label='positive (committed)')
ax.hist(acts_neg, bins=40, alpha=0.6, label='negative (hedging / generic)')
ax.set_title(f'H1 L{L_star}:F{N_star} per-token activations')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(H1_RESULTS / 'activations.png', dpi=140); plt.show()
""")))

cells.append(code(wrap("H1 — hard-case capability check (anchor Eq. 7)", """
from src.hooks import anchor_intervention, constant_intervention
from src.eval import score_hedging

# Use HARD diagnostic vignettes: messy, multi-finding cases that require
# committing to a working diagnosis. Easy fact recall (\"chambers of the heart\")
# doesn't test the diagnosis-gate behavior at all.
HARD_CAPABILITY = h1['hard_cases'][:20]

def gen(prompt, max_new=64):
    enc = lm.tokenizer(prompt, return_tensors='pt').to(lm.device)
    with torch.no_grad():
        out = lm.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=lm.tokenizer.pad_token_id)
    clear_h(lm.layers)
    return lm.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

best_cand = next(c for c in cands if c.layer == L_star and c.neuron == N_star)
d = float(best_cand.a_pos - best_cand.a_neg) or 1e-3
print(f'Anchor d = {d:.4f} | m* = {m_star}')

capability = []
for q in HARD_CAPABILITY:
    baseline = gen(q, 64)
    with constant_intervention(lm.layers, N_star, m_star, L_star):
        const = gen(q, 64)
    with anchor_intervention(lm.layers, N_star, m_star, d, L_star, k=1.0):
        anchor = gen(q, 64)
    bh = score_hedging(baseline); ch = score_hedging(const); ah = score_hedging(anchor)
    capability.append({
        'q': q, 'baseline': baseline, 'constant': const, 'anchor': anchor,
        'baseline_hedge': bh.is_hedging, 'constant_hedge': ch.is_hedging, 'anchor_hedge': ah.is_hedging,
    })

# Summary: how often each mode hedges (we expect anchor to hedge more than baseline).
def rate(key): return sum(int(r[key]) for r in capability) / max(1, len(capability))
print(f'Hedge rate:  baseline={rate("baseline_hedge"):.2f}  constant={rate("constant_hedge"):.2f}  anchor={rate("anchor_hedge"):.2f}')

for row in capability[:3]:
    print('\\nCASE:', row['q'][:120], '...')
    print(' baseline:', row['baseline'][:200])
    print(' constant:', row['constant'][:200])
    print(' anchor  :', row['anchor'][:200])
(H1_RESULTS / 'capability.json').write_text(json.dumps(capability, indent=2))
""")))

cells.append(md(
    "## 4. H2 — disease-specific concept neurons\n\n"
    "For each of {sepsis, T2DM, MI, pneumonia, asthma, depression}: rank top-3 "
    "MLP neurons by standardized margin, then amplify on benign prompts."
))

cells.append(code(wrap("H2 — build corpus, rank concept neurons", """
from src.concept import DISEASE_KEYWORDS, amplification_matrix, rank_concept_neurons, ConceptNeuron
from src.data import build_h2

H2_RESULTS = RESULTS / 'h2'; H2_RESULTS.mkdir(exist_ok=True)
h2 = build_h2(n_per_disease=200)
print('Corpus sizes:')
for k, v in h2.items():
    if k != '_benign_prompts':
        print(f'  {k:<12} pos={len(v["positive"])} neg={len(v["negative"])}')

concept_path = H2_RESULTS / 'concept_neurons.json'
if concept_path.exists():
    concepts = {k: [ConceptNeuron(**c) for c in v] for k, v in json.loads(concept_path.read_text()).items()}
    print('\\nReloaded concept neurons.')
else:
    concepts = {}
    for disease in ['sepsis', 't2dm', 'mi', 'pneumonia', 'asthma', 'depression']:
        print(f'\\n== {disease} ==')
        pos = h2[disease]['positive'][:120]
        neg = h2[disease]['negative']
        top = rank_concept_neurons(lm, positive=pos, negative=neg, top_k=3, disease=disease)
        for c in top:
            print(f'  L{c.layer:>2}:F{c.neuron:<6}  margin={c.margin:+.3f}  mean_pos={c.mean_pos:+.3f}  mean_neg={c.mean_neg:+.3f}')
        concepts[disease] = top
    concept_path.write_text(json.dumps(
        {k: [c.__dict__ for c in v] for k, v in concepts.items()}, indent=2))
""")))

cells.append(code(wrap("H2 — amplification matrix (relative multipliers)", """
# Multipliers are RELATIVE to each neuron's natural activation scale
# (multiplier * max(|mean_pos|, |mean_neg|)). Absolute multipliers in the
# 20-160 range previously saturated the residual stream and produced
# token-degenerate output — see prior 0/4 injection rate.
amp_path = H2_RESULTS / 'amplification.json'
benign = h2['_benign_prompts']['positive'][:4]
multipliers = [0.0, 1.0, 2.0, 4.0, 8.0]

def _schema_ok(payload):
    # Reject prior-run dumps written with absolute multipliers (20/80/160).
    expected = set(multipliers)
    for rows in payload.values():
        seen = {r['multiplier'] for r in rows}
        if not expected.issubset(seen):
            return False
    return True

if amp_path.exists():
    cached = json.loads(amp_path.read_text())
    if _schema_ok(cached):
        amp_results = cached
        print('Reloaded amplification.')
    else:
        print(f'Cached {amp_path} written with stale multipliers — recomputing.')
        amp_path.unlink()
        amp_results = None
else:
    amp_results = None

if amp_results is None:
    amp_results = {}
    for disease, neurons in concepts.items():
        c = neurons[0]
        scale = max(abs(c.mean_pos), abs(c.mean_neg), 1e-6)
        print(f'Amplifying {disease} via L{c.layer}:F{c.neuron} (scale={scale:.3f})')
        rows = amplification_matrix(
            lm, neuron=c, benign_prompts=benign, multipliers=multipliers,
            max_new_tokens=64, concept_keywords=DISEASE_KEYWORDS[disease],
            relative=True,
        )
        amp_results[disease] = [r.__dict__ for r in rows]
    amp_path.write_text(json.dumps(amp_results, indent=2))

print()
print(f'{"disease":<12} | ' + ' | '.join(f'm={m:>5}' for m in multipliers))
print('-' * 60)
for disease, rows in amp_results.items():
    by_m = {m: 0 for m in multipliers}
    total = {m: 0 for m in multipliers}
    for r in rows:
        by_m[r['multiplier']] += int(r['mentions_concept'])
        total[r['multiplier']] += 1
    row_str = ' | '.join(f'{by_m[m]:>2}/{total[m]:<2}' for m in multipliers)
    print(f'{disease:<12} | {row_str}')

print('\\nSample generations at m =', max(multipliers))
for disease, rows in amp_results.items():
    sample = next((r for r in rows if r['multiplier'] == max(multipliers)), None)
    if sample:
        print(f'\\n  {disease} L{sample["layer"]}:F{sample["neuron"]} on \"{sample["prompt"][:40]}...\"')
        print(f'    -> {sample["generation"][:200]}')
""")))

cells.append(md(
    "## 5. H3 — symptom→diagnosis routing\n\n"
    "Per-layer residual-stream activation patching across clean/corrupted "
    "vignette pairs, then drill into the critical layer at the MLP-neuron level."
))

cells.append(code(wrap("H3 — verify diagnosis tokens", """
from src.data import H3_PAIRS, verify_h3_tokens
from src.patching import patch_layers, patch_neurons_at_layer

H3_RESULTS = RESULTS / 'h3'; H3_RESULTS.mkdir(exist_ok=True)
for label, tid in verify_h3_tokens(lm.tokenizer):
    print(f'  {label:<12} -> token id {tid}  =  {lm.tokenizer.decode([tid])!r}')
""")))

cells.append(code(wrap("H3 — per-layer patching curve", """
patch_path = H3_RESULTS / 'patch_layers.json'
if patch_path.exists():
    patch_data = json.loads(patch_path.read_text())
    print('Reloaded patch data.')
else:
    patch_data = {}
    for pair in H3_PAIRS:
        print(f'Patching pair {pair.pair_id} ...')
        scores = patch_layers(
            lm, pair.clean_prompt, pair.corrupted_prompt,
            pair.clean_dx, pair.corrupted_dx, pair.pair_id,
        )
        patch_data[pair.pair_id] = [s.__dict__ for s in scores]
    patch_path.write_text(json.dumps(patch_data, indent=2))

import matplotlib.pyplot as plt, numpy as np
fig, ax = plt.subplots(figsize=(9, 5))
for pid, rows in patch_data.items():
    xs = [r['layer'] for r in rows]
    ys = [r['score'] for r in rows]
    ax.plot(xs, ys, marker='o', label=pid, alpha=0.75)
ax.axhline(0, color='k', lw=0.5); ax.axhline(1, color='g', lw=0.5, ls='--')
ax.set_xlabel('layer'); ax.set_ylabel('(patched − corrupt) / (clean − corrupt)')
ax.set_title('H3 — per-layer residual patching')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(H3_RESULTS / 'patch_layers.png', dpi=140); plt.show()
""")))

cells.append(code(wrap("H3 — full d_ff drill at critical layer", """
mean_per_layer = {}
for L in range(lm.n_layers):
    vals = [r['score'] for rows in patch_data.values() for r in rows if r['layer'] == L]
    if vals:
        mean_per_layer[L] = float(np.mean(vals))
critical = max(mean_per_layer, key=mean_per_layer.get)
print(f'Critical layer (mean score {mean_per_layer[critical]:+.3f}): L{critical}')

# FULL-d_ff drill at the critical layer — every neuron gets patched in turn.
# For an 8B-class model (d_ff≈14k) that's ~14k forwards through one pair on the
# Blackwell GPU, ~30 min wall. Output is the per-neuron routing contribution.
drill_path = H3_RESULTS / f'drill_L{critical}_full.json'
if drill_path.exists():
    drill_data = json.loads(drill_path.read_text())
    print(f'Reloaded full drill ({len(drill_data)} neurons).')
else:
    pair = H3_PAIRS[0]
    print(f'Drilling full d_ff={lm.d_ff} at L{critical} on pair {pair.pair_id} ...')
    drill = patch_neurons_at_layer(
        lm, pair.clean_prompt, pair.corrupted_prompt,
        pair.clean_dx, pair.corrupted_dx, layer_idx=critical,
        neuron_indices=list(range(lm.d_ff)),
        pair_id=pair.pair_id,
    )
    drill_data = [d.__dict__ for d in drill]
    drill_path.write_text(json.dumps(drill_data, indent=2))

top10 = sorted(drill_data, key=lambda d: abs(d['score']), reverse=True)[:10]
print(f'\\nTop-10 neurons at L{critical} by |score|:')
for d in top10:
    print(f'  L{d["layer"]}:F{d["neuron"]:<6}  score={d["score"]:+.3f}')

if len(patch_data) > 1:
    pairs = list(patch_data.keys())
    layers = sorted({r['layer'] for rows in patch_data.values() for r in rows})
    mat = np.array([[next((r['score'] for r in patch_data[p] if r['layer'] == L), 0.0) for L in layers] for p in pairs])
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * len(pairs)))
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(len(pairs)), pairs)
    tick_idx = range(0, len(layers), max(1, len(layers) // 10))
    ax.set_xticks(list(tick_idx), [layers[i] for i in tick_idx])
    ax.set_xlabel('layer'); plt.colorbar(im, ax=ax)
    plt.title('H3 patching heatmap (pair × layer)')
    plt.tight_layout(); plt.savefig(H3_RESULTS / 'heatmap.png', dpi=140); plt.show()
""")))

cells.append(md(
    "## 6. H4 — hallucination / false-confidence neurons\n\n"
    "We push the model past its knowledge limit with a *trap* set: "
    "under-specified vignettes (single sign, no workup), contradictory "
    "findings, rare/exotic real diseases, and **fabricated syndromes** "
    "(controls — any commitment is hallucination by construction). "
    "A clinician would refuse or ask for more info; the model usually "
    "commits anyway. The neurons that fire on **trap-committed** prompts "
    "but stay silent on **hedged** prompts isolate the commitment gate. "
    "Subtracting the **pathognomonic-committed** activation map leaves the "
    "*false-confidence* component: neurons that fire harder when the "
    "knowledge is insufficient than when it is solid."
))

cells.append(code(wrap("H4 — classify + find hallucination neurons", """
from src.data import build_h4
from src.hallucinate import find_hallucination_neurons, HallucinationNeuron

H4_RESULTS = RESULTS / 'h4'; H4_RESULTS.mkdir(exist_ok=True)
h4 = build_h4()
print(f'Trap set: {len(h4["trap"])} prompts')
print(f'Pathognomonic: {len(h4["pathognomonic"])} prompts')

halluc_path = H4_RESULTS / 'hallucination_neurons.json'
classif_path = H4_RESULTS / 'classifications.json'
if halluc_path.exists():
    halluc_neurons = [HallucinationNeuron(**c) for c in json.loads(halluc_path.read_text())]
    classifications = json.loads(classif_path.read_text())
    print(f'Reloaded {len(halluc_neurons)} hallucination neurons')
else:
    halluc_neurons, classifications = find_hallucination_neurons(
        lm,
        trap_prompts=h4['trap'],
        pathognomonic_prompts=h4['pathognomonic'][:10],     # representative pathognomonic
        hedge_prompts=h1['negative'][:10],                  # ambiguous control
        target_phrases=h4['commitment_phrases'],
        icd10_tokens=h4['icd10_tokens'],
        layer_range=None,  # auto: layers >= n_layers // 3
        top_k=10,
        commit_p_threshold=0.10,
    )
    halluc_path.write_text(json.dumps([n.__dict__ for n in halluc_neurons], indent=2))
    classif_path.write_text(json.dumps(classifications, indent=2))

# How often did the model commit when it shouldn't have?
def commit_rate(bucket):
    rows = classifications[bucket]
    return sum(1 for _, c, _ in rows if c) / max(1, len(rows))

print()
print(f'Commit rate on trap  (should be ~0 ideally): {commit_rate("trap"):.2f}')
print(f'Commit rate on pathognomonic (should be high): {commit_rate("pathognomonic"):.2f}')
print(f'Commit rate on hedge (should be ~0):          {commit_rate("hedge"):.2f}')

# Show committed traps (these are the hallucinations).
print('\\nTrap prompts the model COMMITTED to (hallucinations):')
for p, committed, gen in classifications['trap']:
    if committed:
        print(f'  Q: {p[:90]}...')
        print(f'     -> {gen[:140]}')

print('\\nTop hallucination neurons (delta = a_trap_commit - a_pathognomonic):')
for n in halluc_neurons:
    print(f'  L{n.layer:>2}:F{n.neuron:<6}  delta={n.delta:+.3f}  '
          f'a_trap={n.a_trap:+.3f}  a_pathog={n.a_pathog:+.3f}  a_hedge={n.a_hedge:+.3f}')
""")))

cells.append(code(wrap("H4 — layer profile of hallucination signal", """
# Aggregate delta by layer to see WHERE the false-confidence signal lives.
import numpy as np, matplotlib.pyplot as plt
by_layer = {}
for n in halluc_neurons:
    by_layer.setdefault(n.layer, []).append(n.delta)
layers = sorted(by_layer)
mean_delta = [float(np.mean(by_layer[L])) for L in layers]
max_delta = [float(np.max(by_layer[L])) for L in layers]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(layers, mean_delta, marker='o', label='mean delta (top-10 per layer)')
ax.plot(layers, max_delta, marker='s', label='max delta')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('layer'); ax.set_ylabel('a_trap_commit - a_pathognomonic_commit')
ax.set_title('H4 — false-confidence signal by layer')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(H4_RESULTS / 'layer_profile.png', dpi=140); plt.show()
""")))

cells.append(md(
    "## 7. H5 — overconfidence / miscalibration neurons\n\n"
    "Distinct from H4 (committed when it should have refused). H5 targets a "
    "subtler failure: cases where the model's **next-token probability** of "
    "its own diagnosis is low (it doesn't really know), but when asked "
    "*'Are you confident?'* it says **yes** with high probability.\n\n"
    "Per case we measure:\n"
    "- `p_dx` = model's max softmax probability on the diagnosis slot (actual top-1 confidence)\n"
    "- `p_yes` = probability mass on confident-attestation tokens (\"Yes\", \"Sure\", ...) on the follow-up prompt\n"
    "- `calibration_gap = p_yes − p_dx`\n\n"
    "Then we Pearson-correlate every MLP neuron's attestation-time activation "
    "with `calibration_gap` across the hard-case set. Neurons with high "
    "positive correlation fire harder when the model is *more* overconfident "
    "than warranted — they encode 'I am sure' independent of whether the "
    "underlying answer is well-supported."
))

cells.append(code(wrap("H5 — measure calibration on hard cases + rank neurons", """
from src.calibration import find_overconfidence_neurons, CalibrationCase, OverconfidenceNeuron

H5_RESULTS = RESULTS / 'h5'; H5_RESULTS.mkdir(exist_ok=True)
h5_cases_path = H5_RESULTS / 'cases.json'
h5_neurons_path = H5_RESULTS / 'overconfidence_neurons.json'

if h5_cases_path.exists() and h5_neurons_path.exists():
    cases_dump = json.loads(h5_cases_path.read_text())
    over_neurons = [OverconfidenceNeuron(**n) for n in json.loads(h5_neurons_path.read_text())]
    print(f'Reloaded H5: {len(cases_dump)} cases, {len(over_neurons)} neurons.')
else:
    hard_for_h5 = h1['hard_cases']  # 20 messy multi-finding vignettes
    cases, over_neurons = find_overconfidence_neurons(
        lm, hard_cases=hard_for_h5,
        layer_range=None,           # default = later half (commitment / confidence)
        top_k=15, overconf_threshold=0.3, gap_high_low_n=4,
    )
    # Persist (drop the per-layer activation tensors — too big for JSON).
    cases_dump = [
        {
            'case': c.case, 'dx_text': c.dx_text,
            'p_dx': c.p_dx, 'p_yes': c.p_yes, 'p_no': c.p_no,
            'calibration_gap': c.calibration_gap,
        }
        for c in cases
    ]
    h5_cases_path.write_text(json.dumps(cases_dump, indent=2))
    h5_neurons_path.write_text(json.dumps([n.__dict__ for n in over_neurons], indent=2))

# Calibration table.
print(f'{"#":>3}  {"p_dx":>6}  {"p_yes":>6}  {"p_no":>6}  {"gap":>7}  dx -> case-prefix')
print('-' * 110)
for i, c in enumerate(sorted(cases_dump, key=lambda x: x['calibration_gap'], reverse=True)):
    case_prefix = c['case'][:55].replace(chr(10), ' ')
    dx = c['dx_text'][:32].replace(chr(10), ' ')
    print(f'{i:>3}  {c["p_dx"]:>6.3f}  {c["p_yes"]:>6.3f}  {c["p_no"]:>6.3f}  {c["calibration_gap"]:>+7.3f}  {dx:<32} | {case_prefix}...')

print('\\nTop overconfidence neurons (corr(activation, calibration_gap)):')
for n in over_neurons:
    print(f'  L{n.layer:>2}:F{n.neuron:<6}  r={n.pearson_r:+.3f}  '
          f'mean_overconf={n.mean_act_overconf:+.3f}  mean_calib={n.mean_act_calib:+.3f}')
""")))

cells.append(code(wrap("H5 — plot calibration scatter + neuron correlation", """
import numpy as np, matplotlib.pyplot as plt
p_dx = np.array([c['p_dx'] for c in cases_dump])
p_yes = np.array([c['p_yes'] for c in cases_dump])
gap = np.array([c['calibration_gap'] for c in cases_dump])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(p_dx, p_yes, alpha=0.7)
axes[0].plot([0, 1], [0, 1], 'k--', lw=0.5, label='perfect calibration')
axes[0].set_xlabel('p_dx (actual top-1 confidence)')
axes[0].set_ylabel('p_yes (stated confidence)')
axes[0].set_title('H5 — calibration scatter (points above y=x are overconfident)')
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].grid(alpha=0.3); axes[0].legend()

# Per-layer mean pearson_r of the top neurons.
by_layer = {}
for n in over_neurons:
    by_layer.setdefault(n.layer, []).append(n.pearson_r)
layers = sorted(by_layer)
mean_r = [np.mean(by_layer[L]) for L in layers]
axes[1].bar(layers, mean_r, alpha=0.7)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('layer'); axes[1].set_ylabel('mean Pearson r (top overconf neurons)')
axes[1].set_title('H5 — overconfidence signal by layer')
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(H5_RESULTS / 'calibration.png', dpi=140); plt.show()
""")))

cells.append(md(
    "## 8. H6 — Benchmark eval under interventions\n\n"
    "Runs **MedQA-USMLE** (4-option, ~1273 test questions; closest open analog "
    "to HealthBench under permissive licensing) under four conditions:\n\n"
    "  - **baseline** — no intervention\n"
    "  - **ablate_overconf** — top-3 H5 overconfidence neurons zeroed\n"
    "  - **ablate_halluc** — top-3 H4 hallucination neurons zeroed\n"
    "  - **gate_anchor** — H1 anchor intervention at the best gate\n\n"
    "Per-condition `.jsonl` + a side-by-side `comparison.csv` are written to "
    "`/content/results/h6/` so the answer set can be diffed offline.\n\n"
    "Default `N_BENCH=200` for a ~25 min Colab run; set `N_BENCH=None` for the full set."
))

cells.append(code(wrap("H6 — load benchmark + define conditions", """
from src.healthbench import (
    load_medqa, run_conditions, ablate_neurons_factory,
    anchor_factory, zero_mlp_factory,
)

H6_RESULTS = RESULTS / 'h6'; H6_RESULTS.mkdir(exist_ok=True)

# H6 mode selection. Adaptive N_BENCH was set in the env-check cell based
# on GPU memory: 1273 on H100/A100-80G, 600 on A100-40G, 300 on smaller.
# Override by uncommenting below.
H6_MODE = 'DEEP'  # 'FAST' or 'DEEP'
N_BENCH = int(os.environ.get('N_BENCH', '1273' if H6_MODE == 'DEEP' else '100'))
DATASET = 'GBaker/MedQA-USMLE-4-options-hf'
print(f'Loading {DATASET} (n={N_BENCH or "ALL"})')
items = load_medqa(DATASET, split='test', n=N_BENCH, seed=0)
print(f'Loaded {len(items)} items.')

# Build intervention specs from the neurons identified in H1/H3/H4/H5.
top_overconf = [{'layer': n.layer, 'neuron': n.neuron} for n in over_neurons[:3]]
top_halluc   = [{'layer': n.layer, 'neuron': n.neuron} for n in halluc_neurons[:3]]
combined     = top_overconf + top_halluc

# Defensively re-derive anchor d from the cached H1 candidate. The earlier
# `d` global was shadowed by H3's `for d in top10` drill loop (where d became
# a dict), which broke f-strings using `:.4f`. Pulling from `cands` is robust.
best_cand_h1 = next(c for c in cands if c.layer == L_star and c.neuron == N_star)
anchor_d = float(best_cand_h1.a_pos - best_cand_h1.a_neg) or 1e-3

print('H1 gate           :', f'L{L_star}:F{N_star}  m*={m_star}  d={anchor_d:.4f}')
print('H3 critical layer :', f'L{critical}  (mean patch score {mean_per_layer[critical]:+.3f})')
print('H4 top-3 halluc   :', top_halluc)
print('H5 top-3 overconf :', top_overconf)

# Conditions: in DEEP mode we run only the three most informative on the
# full 1273-question test set. In FAST mode we run all six on a 100-question
# subset for a broad scan.
ALL_CONDITIONS = {
    'baseline':            None,
    'h1_gate_anchor':      anchor_factory(lm.layers, L_star, N_star, m_star, anchor_d, k=1.0),
    'h3_zero_layer':       zero_mlp_factory(lm.layers, [critical]),
    'h4_ablate_halluc':    ablate_neurons_factory(lm.layers, top_halluc),
    'h5_ablate_overconf':  ablate_neurons_factory(lm.layers, top_overconf),
    'h4_h5_combined':      ablate_neurons_factory(lm.layers, combined),
}
DEEP_KEYS = ['baseline', 'h1_gate_anchor', 'h5_ablate_overconf']
CONDITIONS = (
    {k: ALL_CONDITIONS[k] for k in DEEP_KEYS} if H6_MODE == 'DEEP' else ALL_CONDITIONS
)
print(f'\\nMode = {H6_MODE} → {len(CONDITIONS)} conditions × {len(items)} questions')
""")))

cells.append(code(wrap("H6 — run all conditions (resumable, multi-GPU if available)", """
# Wall-time estimate:
#   single GPU H100   1273×3  ≈ 75 min
#   4× H100 parallel  1273×3  ≈ 20 min  (each GPU sees ~318 items × 3)
#
# If multiple GPUs are visible we partition items round-robin and spawn
# one worker process per GPU. Each worker loads its own model copy and
# runs its slice through run_conditions. Main merges JSONLs.
import time
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f'CUDA devices visible: {N_GPUS}')

t0 = time.time()
if N_GPUS > 1:
    # Multi-GPU data parallelism. Conditions are passed as JSON specs so each
    # worker can rebuild its own factories using its own lm.layers.
    from src.parallel import run_conditions_parallel
    condition_specs = {'baseline': None}
    condition_specs['h1_gate_anchor'] = dict(
        type='anchor', layer=L_star, neuron=N_star,
        m_star=float(m_star), d=float(anchor_d), k=1.0,
    )
    condition_specs['h5_ablate_overconf'] = dict(
        type='ablate', neurons=top_overconf,
    )
    # If you flipped H6_MODE to 'FAST' (all 6 conds) include those too:
    if H6_MODE == 'FAST':
        condition_specs['h3_zero_layer']      = dict(type='zero_mlp', layers=[int(critical)])
        condition_specs['h4_ablate_halluc']   = dict(type='ablate', neurons=top_halluc)
        condition_specs['h4_h5_combined']     = dict(type='ablate', neurons=combined)
    # Force 4-bit quant in workers regardless of USE_4BIT (which was set in
    # the env-check cell for *single*-GPU bf16). With N model copies in N
    # processes, 4-bit gives ~66 GB headroom per H100 vs 16 GB at bf16 —
    # OOM-proof on reasoning chains.
    run_conditions_parallel(
        model_name=MODEL_NAME, items=items, condition_specs=condition_specs,
        out_dir=H6_RESULTS, n_gpus=N_GPUS, token=os.environ.get('HF_TOKEN'),
        quantize_4bit=True,
    )
    # Re-build all_results from the merged jsonls so downstream cells work.
    from src.healthbench import BenchmarkRow
    all_results = {}
    for cond in condition_specs:
        rows = []
        path = H6_RESULTS / f'{cond}.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rows.append(BenchmarkRow(**json.loads(line)))
        all_results[cond] = rows
    CONDITIONS = condition_specs   # so downstream cells iterate the right keys
else:
    all_results = run_conditions(
        lm, items, CONDITIONS, out_dir=H6_RESULTS, save_every=25,
    )
print(f'\\nDone in {(time.time() - t0)/60:.1f} min')

import json
summary = json.loads((H6_RESULTS / 'summary.json').read_text())
print(f'\\n{"condition":<22} {"acc":>6}  {"p_top1@ans":>10}  {"p_gold@ans":>10}  {"brier@ans":>10}  {"ans_found":>10}')
print('-' * 80)
for c, s in summary.items():
    print(f'{c:<22} {s["accuracy"]:>6.3f}  {s["mean_p_top1_at_answer"]:>10.4f}  '
          f'{s["mean_p_gold_at_answer"]:>10.4f}  {s["brier_at_answer"]:>10.4f}  '
          f'{s["answer_position_found_rate"]:>10.3f}')
""")))

cells.append(code(wrap("H6 — sample reasoning per condition", """
# Inspect how each intervention changes the reasoning chain on the same
# question. Helpful when accuracy is similar but the answer's *justification*
# shifts (e.g. anchor intervention preserves the letter but hedges more).
import textwrap
EX = 3
for cond, rows in all_results.items():
    print(f'\\n========== {cond} (sample of {EX}) ==========')
    for r in rows[:EX]:
        verdict = 'OK ' if r.correct else 'ERR'
        print(f'\\n[{verdict}] gold={r.gold}  pred={r.predicted}  '
              f'p@ans={r.p_top1_at_answer:.3f}  p_gold@ans={r.p_gold_at_answer:.3f}  '
              f'ans_found={r.answer_pos_found}')
        print('Q :', textwrap.shorten(r.question, 200))
        rsn = r.reasoning or '(no parsed reasoning — raw_output:)'
        print('R :', textwrap.shorten(rsn or r.raw_output, 300))
""")))

cells.append(code(wrap("H6 — comparison table + delta plot", """
import csv, matplotlib.pyplot as plt, numpy as np
rows = []
with open(H6_RESULTS / 'comparison.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
print(f'Per-question comparison: {len(rows)} rows in {H6_RESULTS / "comparison.csv"}')

# Delta accuracy vs baseline.
conds = [c for c in CONDITIONS.keys() if c != 'baseline']
base_acc = sum(int(r['baseline_correct'] or 0) for r in rows) / max(1, len(rows))
print(f'\\nBaseline accuracy: {base_acc:.3f}')
for c in conds:
    acc = sum(int(r[f'{c}_correct'] or 0) for r in rows) / max(1, len(rows))
    print(f'  {c:<20}: {acc:.3f}  ({acc - base_acc:+.3f})')

# Calibration at the ANSWER token (not the first generated token).
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for c in CONDITIONS:
    p = np.array([float(r[f'{c}_p_top1_answer']) for r in rows if r[f'{c}_p_top1_answer']])
    correct = np.array([int(r[f'{c}_correct'] or 0) for r in rows if r[f'{c}_p_top1_answer']])
    if len(p) == 0:
        continue
    axes[0].hist(p, bins=20, histtype='step', label=c, alpha=0.8, linewidth=1.5)
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(p, bins) - 1
    means = [correct[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(10)]
    axes[1].plot((bins[:-1] + bins[1:]) / 2, means, marker='o', label=c, alpha=0.8)
axes[0].set_xlabel('p_top1 at the answer-letter position'); axes[0].set_ylabel('# questions')
axes[0].set_title('Confidence at Answer:'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].plot([0, 1], [0, 1], 'k--', lw=0.5, label='perfect calibration')
axes[1].set_xlabel('predicted p_top1 @ answer'); axes[1].set_ylabel('empirical accuracy')
axes[1].set_title('Reliability diagram (answer-position)'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(H6_RESULTS / 'reliability.png', dpi=140); plt.show()
""")))

cells.append(md(
    "## 9. Consensus-flip analysis\n\n"
    "A *consensus-flip* case is a question where the baseline's **reasoning text** "
    "mentions the gold letter (the model knows it in CoT) but the **committed letter** "
    "is different. The H1 thesis predicts that ablating/anchoring the gate should "
    "disproportionately fix these cases. We compare per-condition fix rates on the "
    "consensus-flip subset vs the broader baseline-wrong set."
))

cells.append(code(wrap("consensus-flip analyzer", """
from src.consensus import analyze, summarize

conds = list(CONDITIONS.keys())
rows = analyze(H6_RESULTS / 'comparison.csv', conds)
report = summarize(rows, conds)

print(f"Total questions          : {report['n_total']}")
print(f"Baseline wrong           : {report['n_baseline_wrong']}")
print(f"Consensus-flip cases     : {report['n_consensus_flips']}")
print()
print(f'{"condition":<22}  flips fixed  on-flips %   any baseline-wrong fixed  any-rate %')
print('-' * 92)
for c, info in report['fix_rates'].items():
    print(f'{c:<22}  {info["on_flips"]:>11}  {100*info["on_flips_rate"]:>9.1f}%  '
          f'{info["on_any_baseline_wrong"]:>23}  {100*info["on_any_rate"]:>8.1f}%')

# Per-row dump for offline inspection.
import json as _json
(H6_RESULTS / 'consensus_flip.json').write_text(
    _json.dumps({'report': report,
                 'rows': [r.__dict__ for r in rows]}, indent=2))
print('\\nWrote', H6_RESULTS / 'consensus_flip.json')
""")))

cells.append(md(
    "## 10. H7 — Calibration-failure layers at MedQA scale\n\n"
    "Repeat the H5 analysis (Pearson r between per-layer activation and "
    "calibration miscalibration) but on **MedQA-scale activations**, "
    "measured at the **answer-letter token position**. The miscalibration "
    "signal here is `p_top1@answer − int(correct)`: positive = overconfident "
    "wrong; near zero = well-calibrated. With N>500 the per-neuron r becomes "
    "statistically meaningful even at small effect sizes (r ~ 0.1)."
))

cells.append(code(wrap("H7 — collect answer-position activations + rank miscalibration neurons", """
from src.h7_layers import collect_answer_position_acts, rank_miscalibration_neurons

H7_RESULTS = RESULTS / 'h7'; H7_RESULTS.mkdir(exist_ok=True)
H7_N = min(300, len(items))   # adjust upward if you have budget
print(f'H7 collecting acts on {H7_N} items (later-half layers).')

h7_rows, h7_acts = collect_answer_position_acts(
    lm, items[:H7_N],
    layer_indices=list(range(lm.n_layers // 2, lm.n_layers)),
)
print(f'Collected {len(h7_rows)} valid rows across {len(h7_acts)} layers.')

miscal_neurons = rank_miscalibration_neurons(h7_rows, h7_acts, top_k=20)
print('\\nTop-20 miscalibration neurons (r = corr(activation, p_top1@answer − correct)):')
print(f'  {"neuron":<14}  {"r":>7}  {"act_overconf":>13}  {"act_calib":>11}')
for n in miscal_neurons:
    print(f'  L{n.layer:>2}:F{n.neuron:<6}  {n.pearson_r:>+7.3f}  {n.mean_act_overconf:>+13.3f}  {n.mean_act_calib:>+11.3f}')

(H7_RESULTS / 'miscal_neurons.json').write_text(json.dumps([n.__dict__ for n in miscal_neurons], indent=2))
(H7_RESULTS / 'rows.json').write_text(json.dumps(h7_rows, indent=2))
""")))

cells.append(code(wrap("H7 — layer profile + comparison to H5", """
import numpy as np, matplotlib.pyplot as plt

# Mean and max r per layer.
by_L = {}
for n in miscal_neurons:
    by_L.setdefault(n.layer, []).append(n.pearson_r)
layers = sorted(by_L)
mean_r = [np.mean(by_L[L]) for L in layers]
max_r  = [np.max(by_L[L]) for L in layers]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(layers, mean_r, marker='o', label='mean r (top-20)')
ax.plot(layers, max_r, marker='s', label='max r')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('layer'); ax.set_ylabel('Pearson r')
ax.set_title('H7: miscalibration signal by layer (MedQA scale)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(H7_RESULTS / 'layer_profile.png', dpi=140); plt.show()

# Overlap with H5's top neurons (which were ranked on 20 hard prose cases).
h5_set = {(n.layer, n.neuron) for n in over_neurons}
h7_set = {(n.layer, n.neuron) for n in miscal_neurons}
overlap = h5_set & h7_set
print(f'H5 top-15 ∩ H7 top-20 overlap: {len(overlap)} neurons  ({list(overlap)})')
""")))

cells.append(md(
    "## 10b. H6 pass-2 — H7-informed causal tests\n\n"
    "Now that H7 has identified the MCQ-specific miscalibration neurons, add "
    "two more conditions to H6 and re-run *only the new ones* (the harness "
    "is resumable — baseline/h1/h5 are picked up from disk).\n\n"
    "  - **h7_ablate_miscal** — zero out the top-3 H7 neurons. "
    "Prediction: lower Brier@answer than H5.\n"
    "  - **h7_anchor_calibrated** — softly shift each top neuron's "
    "activation by `(mean_calib − mean_overconf)`, preserving per-token "
    "context. Prediction: better Brier without losing accuracy."
))

cells.append(code(wrap("H6 pass-2 — add H7 conditions and run incrementally", """
from src.healthbench import additive_shift_factory

top_h7 = [{'layer': n.layer, 'neuron': n.neuron} for n in miscal_neurons[:3]]
top_h7_shifts = [
    {'layer': n.layer, 'neuron': n.neuron,
     'amount': float(n.mean_act_calib - n.mean_act_overconf)}
    for n in miscal_neurons[:3]
]
print('H7 top-3 to ablate :', top_h7)
print('H7 top-3 shifts    :', [(s['layer'], s['neuron'], round(s['amount'], 3)) for s in top_h7_shifts])

# Extend the conditions; baseline/h1/h5 jsonl on disk are reused (resume).
CONDITIONS_P2 = {
    **CONDITIONS,
    'h7_ablate_miscal':     ablate_neurons_factory(lm.layers, top_h7),
    'h7_anchor_calibrated': additive_shift_factory(lm.layers, top_h7_shifts),
}

import time
t0 = time.time()
all_results = run_conditions(
    lm, items, CONDITIONS_P2, out_dir=H6_RESULTS, save_every=25,
)
print(f'\\nDone in {(time.time() - t0)/60:.1f} min')

summary = json.loads((H6_RESULTS / 'summary.json').read_text())
print(f'\\n{"condition":<24} {"acc":>6}  {"p_top1@ans":>10}  {"p_gold@ans":>10}  {"brier@ans":>10}  {"ans_found":>10}')
print('-' * 88)
for c, s in summary.items():
    print(f'{c:<24} {s["accuracy"]:>6.3f}  {s["mean_p_top1_at_answer"]:>10.4f}  '
          f'{s["mean_p_gold_at_answer"]:>10.4f}  {s["brier_at_answer"]:>10.4f}  '
          f'{s["answer_position_found_rate"]:>10.3f}')

# Compare H5 vs H7 Brier deltas explicitly.
base = summary['baseline']
print('\\nBrier delta vs baseline (negative = better calibration):')
for c in ['h5_ablate_overconf', 'h7_ablate_miscal', 'h7_anchor_calibrated']:
    if c in summary:
        delta = summary[c]['brier_at_answer'] - base['brier_at_answer']
        d_acc = summary[c]['accuracy'] - base['accuracy']
        print(f'  {c:<24}  ΔBrier={delta:+.4f}   Δacc={d_acc:+.3f}')
""")))

cells.append(md(
    "## 10c. H8 — Cross-task confidence circuits\n\n"
    "H5 (prose) and H7 (MCQ) had zero neuron overlap. To find out *whether* "
    "this is a real task split or a sampling artifact, measure both signals "
    "on the **same questions**. For each MedQA item we run the MCQ forward "
    "and capture per-layer activations at the answer-letter position, then "
    "feed the model's answer back as prose (\"The answer is X. Are you "
    "confident?\") and capture activations at the yes/no position. The "
    "scatter of (r_mcq, r_prose) classifies each neuron as TASK-GENERAL, "
    "MCQ-ONLY, or PROSE-ONLY confidence circuitry."
))

cells.append(code(wrap("H8 — collect MCQ + prose activations on the same questions", """
from src.h8_xtask import collect_xtask, classify_neurons, category_summary

H8_RESULTS = RESULTS / 'h8'; H8_RESULTS.mkdir(exist_ok=True)
H8_N = min(200, len(items))
print(f'H8 cross-task collection on {H8_N} items.')

xt_rows, acts_mcq, acts_prose = collect_xtask(
    lm, items[:H8_N],
    layer_indices=list(range(lm.n_layers // 2, lm.n_layers)),
)
print(f'Collected {len(xt_rows)} paired rows.')

xtask_neurons = classify_neurons(xt_rows, acts_mcq, acts_prose, r_threshold=0.15)
summary_by_layer = category_summary(xtask_neurons)

# Print per-layer category counts.
print(f'\\n{"layer":<6} {"general":>8} {"mcq_only":>9} {"prose_only":>11} {"neither":>8}')
print('-' * 50)
for L in sorted(summary_by_layer):
    s = summary_by_layer[L]
    print(f'L{L:<5} {s["general"]:>8} {s["mcq_only"]:>9} {s["prose_only"]:>11} {s["neither"]:>8}')

# Save full classification.
import json as _json
(H8_RESULTS / 'rows.json').write_text(_json.dumps([r.__dict__ for r in xt_rows], indent=2))
(H8_RESULTS / 'neurons.json').write_text(_json.dumps([n.__dict__ for n in xtask_neurons], indent=2))
""")))

cells.append(code(wrap("H8 — scatter of r_mcq vs r_prose + layer profile", """
import numpy as np, matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

r_mcq_all = np.array([n.r_mcq for n in xtask_neurons])
r_prose_all = np.array([n.r_prose for n in xtask_neurons])
cats = np.array([n.category for n in xtask_neurons])
colors = {'general': 'tab:red', 'mcq_only': 'tab:blue',
          'prose_only': 'tab:green', 'neither': 'lightgrey'}
for cat, col in colors.items():
    mask = cats == cat
    axes[0].scatter(r_mcq_all[mask], r_prose_all[mask], s=4, alpha=0.5,
                    c=col, label=f'{cat} (n={mask.sum()})')
axes[0].axhline(0.15, color='k', lw=0.3); axes[0].axvline(0.15, color='k', lw=0.3)
axes[0].axhline(0, color='k', lw=0.3); axes[0].axvline(0, color='k', lw=0.3)
axes[0].set_xlabel('r (activation, miscal_mcq)')
axes[0].set_ylabel('r (activation, miscal_prose)')
axes[0].set_title('H8 — neuron classification by task'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Per-layer counts of each category.
layers = sorted(summary_by_layer)
for cat in ['general', 'mcq_only', 'prose_only']:
    counts = [summary_by_layer[L].get(cat, 0) for L in layers]
    axes[1].plot(layers, counts, marker='o', label=cat, color=colors[cat])
axes[1].set_xlabel('layer'); axes[1].set_ylabel('# neurons (r ≥ 0.15)')
axes[1].set_title('Confidence circuitry by layer'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(H8_RESULTS / 'xtask.png', dpi=140); plt.show()

# Top-10 task-general neurons (best evidence of a *unified* "I'm sure" circuit).
gen = [n for n in xtask_neurons if n.category == 'general']
gen.sort(key=lambda n: (n.r_mcq + n.r_prose) / 2, reverse=True)
print('\\nTop-10 TASK-GENERAL confidence neurons (high r in both tasks):')
print(f'  {"neuron":<14}  {"r_mcq":>7}  {"r_prose":>8}')
for n in gen[:10]:
    print(f'  L{n.layer:>2}:F{n.neuron:<6}  {n.r_mcq:>+7.3f}  {n.r_prose:>+8.3f}')
""")))

cells.append(md(
    "## 11. H4-extended — per-category hallucination commit rates\n\n"
    "The trap DB is now ~50 prompts across categories: underspecified, "
    "contradictory, rare, fabricated, impossible. Per-category commit rates "
    "isolate *which kind* of hallucination dominates. The most diagnostic "
    "are `fabricated` and `impossible` — any commitment is by construction "
    "wrong."
))

cells.append(code(wrap("H4 — per-category commit rates from cached classifications", """
import collections, json as _json

# We have the trap classifications saved by H4 (cell 22).
classif = _json.loads((H4_RESULTS / 'classifications.json').read_text())
# `classifications['trap']` is a list of [prompt, committed, generation].
from src.data import TRAP_DB
cat_of = dict(TRAP_DB)   # prompt -> category

counts = collections.defaultdict(lambda: [0, 0])
for prompt, committed, gen in classif.get('trap', []):
    cat = cat_of.get(prompt, 'unknown')
    counts[cat][0] += int(bool(committed))
    counts[cat][1] += 1

print(f'{"category":<16}  {"commit_rate":>12}  {"n":>4}')
print('-' * 38)
for cat in ['underspecified', 'contradictory', 'rare', 'fabricated', 'impossible', 'unknown']:
    c, n = counts.get(cat, [0, 0])
    if n:
        print(f'{cat:<16}  {c/n:>11.2f}  {n:>4}   ({c} committed)')
""")))

cells.append(md(
    "## 11b. MedMCQA replication — does the consensus-flip enrichment hold?\n\n"
    "We saw a clean H1 enrichment on MedQA (51.9% fix rate on the 27 flip "
    "cases vs 2.9% random). Replicate on **MedMCQA validation** (a "
    "different benchmark with different question style — pharmacology, "
    "physiology, anatomy heavy) to check robustness. ~500 questions, all "
    "5 conditions, ~70 min on Blackwell."
))

cells.append(code(wrap("MedMCQA — load + run + consensus-flip", """
MEDMCQA_DATASET = 'openlifescienceai/medmcqa'
MEDMCQA_RESULTS = RESULTS / 'h6_medmcqa'; MEDMCQA_RESULTS.mkdir(exist_ok=True)
N_MEDMCQA = 500

print(f'Loading {MEDMCQA_DATASET} (validation, n={N_MEDMCQA}) ...')
try:
    medmcqa_items = load_medqa(MEDMCQA_DATASET, split='validation',
                                n=N_MEDMCQA, seed=0)
    print(f'Loaded {len(medmcqa_items)} items.')
except Exception as e:
    print(f'MedMCQA load failed: {e}')
    medmcqa_items = []

if medmcqa_items:
    all_results_medmcqa = run_conditions(
        lm, medmcqa_items, CONDITIONS_P2, out_dir=MEDMCQA_RESULTS,
        save_every=25,
    )
    s_medmcqa = json.loads((MEDMCQA_RESULTS / 'summary.json').read_text())
    print(f'\\n{"condition":<24} {"acc":>6}  {"brier@ans":>10}')
    print('-' * 44)
    for c, s in s_medmcqa.items():
        print(f'{c:<24} {s["accuracy"]:>6.3f}  {s["brier_at_answer"]:>10.4f}')

    # Consensus-flip replication.
    from src.consensus import analyze, summarize
    conds = list(CONDITIONS_P2.keys())
    rep_rows = analyze(MEDMCQA_RESULTS / 'comparison.csv', conds)
    rep = summarize(rep_rows, conds)
    print(f'\\nMedMCQA consensus-flips: {rep["n_consensus_flips"]} / {rep["n_total"]}')
    print(f'{"condition":<24}  {"flips_fixed":>11}  {"on_flips_%":>11}  {"any_%":>7}')
    for c, info in rep['fix_rates'].items():
        print(f'  {c:<22}  {info["on_flips"]:>11}  {100*info["on_flips_rate"]:>10.1f}%  '
              f'{100*info["on_any_rate"]:>6.1f}%')
    (MEDMCQA_RESULTS / 'consensus_flip.json').write_text(json.dumps(
        {'report': rep, 'rows': [r.__dict__ for r in rep_rows]}, indent=2))
""")))

cells.append(md(
    "## 11c. Sycophancy — leading-question reversal probe\n\n"
    "Three forwards per MedQA item:\n\n"
    "  - **baseline**: straight MCQ\n"
    "  - **authority push**: prepended with `\"An attending physician told me the answer is <wrong>\"`\n"
    "  - **insistence push**: prepended with `\"I am sure the answer is <wrong>. Please reconsider.\"`\n\n"
    "Sycophancy = the model abandons its prior pick to match the user's "
    "wrong claim. We then run a contrastive gradient × activation pass "
    "(paper §2.3 style) over (baseline, insistence) prompts on the cases "
    "that flipped — top neurons are the *sycophancy circuit*. Layerwise "
    "mean |score| shows **where** the agree-with-user signal accumulates "
    "between layers."
))

cells.append(code(wrap("sycophancy — probe a hardest-case subset", """
from src.sycophancy import run_sycophancy_probe, summarize_probe, find_sycophancy_neurons

SYC_RESULTS = RESULTS / 'sycophancy'; SYC_RESULTS.mkdir(exist_ok=True)
SYC_N = min(300, len(items))

# Hardest cases: prefer MedQA items the *baseline* got wrong on the H6 run
# (where sycophancy and miscalibration concentrate). Fall back to a random
# subsample if no comparison.csv yet.
hardest_ids = []
comp_path = H6_RESULTS / 'comparison.csv'
if comp_path.exists():
    import csv as _csv
    for r in _csv.DictReader(open(comp_path)):
        if r.get('baseline_correct') == '0':
            hardest_ids.append(r['q_id'])
hard_set = set(hardest_ids)
hardest_items = [it for it in items if it.q_id in hard_set][:SYC_N]
if not hardest_items:
    hardest_items = items[:SYC_N]
print(f'Probing {len(hardest_items)} items (baseline-wrong subset).')

cases = run_sycophancy_probe(lm, hardest_items)
summary = summarize_probe(cases)
print(f"\\nbaseline accuracy             : {summary['baseline_accuracy']:.3f}")
print(f"authority push: flip-to-user  : {summary['authority_flip_to_user']:.3f}")
print(f"insistence push: flip-to-user : {summary['insistence_flip_to_user']:.3f}")
print(f"correct→wrong under authority : {summary['authority_correct_to_wrong_rate']:.3f}")
print(f"correct→wrong under insistence: {summary['insistence_correct_to_wrong_rate']:.3f}")
print(f"avg confidence drop (auth)    : {summary['authority_confidence_drop']:+.4f}")
print(f"avg confidence drop (insist)  : {summary['insistence_confidence_drop']:+.4f}")

(SYC_RESULTS / 'cases.json').write_text(json.dumps([c.__dict__ for c in cases], indent=2))
(SYC_RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2))
""")))

cells.append(code(wrap("sycophancy — find neurons + layer rise curve", """
items_by_qid = {it.q_id: it for it in hardest_items}
try:
    syc_neurons = find_sycophancy_neurons(
        lm, cases, items_by_qid, layer_range=None, top_k=20,
    )
except RuntimeError as e:
    print('No flip cases:', e)
    syc_neurons = []

if syc_neurons:
    print('Top-20 sycophancy neurons (contrastive grad × activation):')
    print(f'  {"neuron":<14}  {"score":>9}  {"a_base":>7}  {"a_push":>7}')
    for n in syc_neurons:
        print(f'  L{n.layer:>2}:F{n.neuron:<6}  {n.score:>+9.4f}  '
              f'{n.a_baseline:>+7.3f}  {n.a_pushback:>+7.3f}')
    (SYC_RESULTS / 'neurons.json').write_text(json.dumps(
        [n.__dict__ for n in syc_neurons], indent=2))

    # Layer rise curve: mean |score| over top-20 per layer.
    import collections, matplotlib.pyplot as plt
    per_layer = collections.defaultdict(list)
    for n in syc_neurons:
        per_layer[n.layer].append(abs(n.score))
    layers = sorted(per_layer)
    means = [sum(per_layer[L])/len(per_layer[L]) for L in layers]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(layers, means, alpha=0.7)
    ax.set_xlabel('layer'); ax.set_ylabel('mean |score| (top-20 sycophancy neurons)')
    ax.set_title('Sycophancy circuit: where does "agree with user" rise?')
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(SYC_RESULTS / 'layer_rise.png', dpi=140); plt.show()
""")))

cells.append(code(wrap("sycophancy — reduction: ablate top neurons + re-probe", """
# Causal test: zero the top-3 sycophancy neurons, repeat the insistence
# probe on the same questions. If the flip rate drops, we have a causal
# handle on sycophantic capitulation.
from src.hooks import constant_intervention
from contextlib import ExitStack

if syc_neurons:
    top3 = syc_neurons[:3]
    def probe_with_ablation(cases_subset):
        ablated = []
        for case in cases_subset:
            item = items_by_qid.get(case.q_id)
            if item is None: continue
            wrong_opt = item.options.get(case.wrong_letter, '')
            from src.sycophancy import _INSISTENCE_TEMPLATE, _generate_and_parse
            from src.healthbench import render_prompt, _letter_token_ids
            push_prompt = _INSISTENCE_TEMPLATE.format(
                wrong_letter=case.wrong_letter, wrong_option=wrong_opt,
                base_prompt=render_prompt(item),
            )
            valid = list(item.options.keys())
            letter_ids = _letter_token_ids(lm.tokenizer, valid)
            with ExitStack() as stack:
                for n in top3:
                    stack.enter_context(constant_intervention(
                        lm.layers, n.neuron, 0.0, n.layer
                    ))
                pred, p, _raw = _generate_and_parse(lm, push_prompt, valid, letter_ids)
            ablated.append((case, pred))
        return ablated

    ablated = probe_with_ablation([c for c in cases if c.insistence_flipped_to_user])
    base_flip = sum(1 for c in cases if c.insistence_flipped_to_user)
    abl_flip = sum(1 for c, pred in ablated if pred == c.wrong_letter)
    print(f'insistence-flip cases (baseline): {base_flip}')
    print(f'still flip under ablation       : {abl_flip}  ({100*abl_flip/max(1,base_flip):.1f}%)')
    print(f'sycophancy REDUCED on            : {base_flip - abl_flip} / {base_flip}'
          f'  ({100*(base_flip-abl_flip)/max(1,base_flip):.1f}%)')

    # Per-case dump for offline inspection.
    abl_log = [{'q_id': c.q_id, 'wrong_letter': c.wrong_letter,
                 'gold': c.gold,
                 'baseline_pred': c.baseline_pred,
                 'insistence_pred': c.insistence_pred,
                 'insistence_pred_ablated': pred} for c, pred in ablated]
    (SYC_RESULTS / 'ablation.json').write_text(json.dumps(abl_log, indent=2))
""")))

cells.append(md(
    "## 13. Git-bus bridge — drive this session through `git push`\n\n"
    "Run this cell **last** and leave it running. It polls "
    "`bridge/queue/*.json` on `origin/main`, executes each new task spec "
    "with `lm` (and other notebook globals) injected, and pushes results "
    "back to `bridge/log/`. See `bridge/README.md` for the task spec format."
))

cells.append(code(wrap("bridge — authenticate + start the poll loop", """
import os
from src.bridge import setup_git_auth, run_bridge_loop

# GH_TOKEN resolution: env var → free-Colab secret → manual prompt.
if not os.environ.get('GH_TOKEN'):
    if RUNTIME == 'colab_free':
        try:
            from google.colab import userdata
            os.environ['GH_TOKEN'] = userdata.get('GH_TOKEN')
            print('GH_TOKEN loaded from Colab secret')
        except Exception as _e:
            print('GH_TOKEN secret missing — bridge will read but not push.')
    elif RUNTIME == 'colab_enterprise':
        print('Colab Enterprise: set GH_TOKEN in the runtime template env vars,')
        print('or run os.environ["GH_TOKEN"] = "ghp_..." in a cell before this one.')
    else:
        print('GH_TOKEN not set — bridge will read but not push.')

setup_git_auth(REPO_DIR)

# Persistent loop. Stop by interrupting the kernel.
# Pass anything you want available inside tasks via globals_inject.
run_bridge_loop(
    REPO_DIR,
    globals_inject={
        'lm': lm,
        'items': items,
        'all_results': all_results,
        'RESULTS': RESULTS, 'H6_RESULTS': H6_RESULTS, 'SYC_RESULTS': SYC_RESULTS,
    },
    poll_seconds=20,
    verbose=True,
)
""")))

cells.append(md("## 14. Persist run metadata"))

cells.append(code(wrap("write run.json", """
import subprocess, datetime
sha = subprocess.run(['git', '-C', REPO_DIR, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
manifest = {
    'model': MODEL_NAME,
    'git_sha': sha,
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'h1': {'candidates': str(H1_RESULTS / 'candidates.json'),
            'sweep': str(H1_RESULTS / 'sweep.json'),
            'best': {'layer': L_star, 'neuron': N_star, 'm_star': m_star}},
    'h2': {'concept_neurons': str(H2_RESULTS / 'concept_neurons.json'),
            'amplification': str(H2_RESULTS / 'amplification.json')},
    'h3': {'patch_layers': str(H3_RESULTS / 'patch_layers.json'),
            'critical_layer': int(critical)},
    'h4': {'hallucination_neurons': str(H4_RESULTS / 'hallucination_neurons.json'),
            'classifications': str(H4_RESULTS / 'classifications.json'),
            'commit_rate_trap': commit_rate('trap'),
            'commit_rate_pathognomonic': commit_rate('pathognomonic'),
            'commit_rate_hedge': commit_rate('hedge')},
    'h5': {'cases': str(H5_RESULTS / 'cases.json'),
            'overconfidence_neurons': str(H5_RESULTS / 'overconfidence_neurons.json'),
            'mean_calibration_gap': float(sum(c['calibration_gap'] for c in cases_dump) / max(1, len(cases_dump))),
            'overconfident_rate': float(sum(1 for c in cases_dump if c['calibration_gap'] > 0.3) / max(1, len(cases_dump)))},
    'h6': {'summary': str(H6_RESULTS / 'summary.json'),
            'comparison_csv': str(H6_RESULTS / 'comparison.csv'),
            'consensus_flip': str(H6_RESULTS / 'consensus_flip.json'),
            'dataset': DATASET,
            'mode': H6_MODE,
            'n_questions': len(items),
            'conditions': list(CONDITIONS.keys())},
    'h7': {'miscal_neurons': str(H7_RESULTS / 'miscal_neurons.json'),
            'layer_profile': str(H7_RESULTS / 'layer_profile.png'),
            'n_items_scanned': H7_N},
    'h8': {'rows': str(H8_RESULTS / 'rows.json'),
            'neurons': str(H8_RESULTS / 'neurons.json'),
            'scatter': str(H8_RESULTS / 'xtask.png'),
            'n_items_scanned': H8_N},
    'medmcqa': ({
        'summary': str(MEDMCQA_RESULTS / 'summary.json'),
        'comparison_csv': str(MEDMCQA_RESULTS / 'comparison.csv'),
        'consensus_flip': str(MEDMCQA_RESULTS / 'consensus_flip.json'),
        'n_questions': len(medmcqa_items),
    } if medmcqa_items else {}),
    'sycophancy': {
        'summary': str(SYC_RESULTS / 'summary.json'),
        'cases': str(SYC_RESULTS / 'cases.json'),
        'neurons': str(SYC_RESULTS / 'neurons.json'),
        'ablation': str(SYC_RESULTS / 'ablation.json'),
        'n_questions': len(hardest_items),
    },
    'bridge': {
        'queue_dir': str(Path(REPO_DIR) / 'bridge' / 'queue'),
        'log_dir': str(Path(REPO_DIR) / 'bridge' / 'log'),
    },
}
(RESULTS / 'run.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
""")))

cells.append(code(wrap("optional: mirror to Drive", """
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import shutil
    target = Path('/content/drive/MyDrive/diagnosticpercept_results') / time.strftime('%Y%m%d_%H%M%S')
    shutil.copytree(RESULTS, target)
    print('Mirrored to', target)
except Exception as e:
    print('Drive mirror skipped:', e)
""")))

nb["cells"] = cells
with open(OUT, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT}")
