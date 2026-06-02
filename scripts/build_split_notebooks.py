"""Generate per-phase notebooks for Colab Enterprise from the monolith.

The monolith (``notebooks/diagnostic_percept.ipynb``) runs the whole pipeline
in one session. On Colab Enterprise each phase is better run in its OWN
runtime so a disconnect in the expensive H6/H7 stage doesn't force re-running
the cheap H1-H5 discovery. This script slices the monolith's *already-correct*
cells into five phase notebooks and bolts on:

  * a shared persistence preamble (restore ``results/`` from GCS/Drive) so a
    fresh runtime sees the prior phase's artifacts (``src.persist``), and
  * a discovery checkpoint hand-off (``src.checkpoint``) so the benchmark /
    scale / sycophancy phases reload the H1-H5 neuron coordinates without a
    GPU pass through discovery.

Phases
------
  00_setup_check  — install + GPU + disk + model load + hook sanity. Run once
                    to confirm the runtime is healthy before a long job.
  01_discovery    — H1·H2·H3·H4·H4ext·H5 → writes results/discovery.json
  02_benchmark    — restore → H6 (6 conditions, multi-GPU) + consensus-flip
  03_scale        — restore → H7 · H6 pass-2 · H8 · MedMCQA replication
  04_sycophancy   — restore → sycophancy probe · neurons · reduction

Reusing the monolith cells verbatim (rather than re-authoring phase logic)
means the split notebooks inherit every fix already verified in ``src/`` and
the monolith. Only the persistence + checkpoint glue is new, and that glue is
unit-tested in ``tests/test_persist.py`` / ``tests/test_checkpoint.py``.

Run with::

    python scripts/build_notebook.py        # regenerate the monolith first
    python scripts/build_split_notebooks.py  # then the split set
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
MONOLITH = ROOT / "notebooks" / "diagnostic_percept.ipynb"
OUT_DIR = ROOT / "notebooks" / "split"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/ArioMoniri/diagnosticpercept.git"


# ---------------------------------------------------------------------------
# Load the monolith + index cells
# ---------------------------------------------------------------------------

_mono = nbf.read(MONOLITH, as_version=4)
_CELLS = _mono.cells


def _label_of(cell) -> str:
    if cell.cell_type != "code":
        return ""
    m = re.search(r"# === (.+?) ===", cell.source)
    return m.group(1) if m else ""


_BY_LABEL: Dict[str, nbf.NotebookNode] = {}
for c in _CELLS:
    lab = _label_of(c)
    if lab:
        _BY_LABEL[lab] = c


def code_by_label(label: str) -> nbf.NotebookNode:
    if label not in _BY_LABEL:
        raise KeyError(f"no code cell labelled {label!r} in the monolith. "
                       f"Re-run scripts/build_notebook.py first.")
    return nbf.v4.new_code_cell(_BY_LABEL[label].source)


def md_containing(substr: str) -> nbf.NotebookNode:
    for c in _CELLS:
        if c.cell_type == "markdown" and substr in c.source:
            return nbf.v4.new_markdown_cell(c.source)
    raise KeyError(f"no markdown cell containing {substr!r}")


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s)


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip("\n"))


# Shared setup cells = monolith index 1..13 (## Setup … hook sanity), cloned.
def setup_cells() -> List[nbf.NotebookNode]:
    out = []
    for c in _CELLS[1:14]:
        if c.cell_type == "markdown":
            out.append(nbf.v4.new_markdown_cell(c.source))
        else:
            out.append(nbf.v4.new_code_cell(c.source))
    return out


# ---------------------------------------------------------------------------
# Glue cells (new; the only authored logic)
# ---------------------------------------------------------------------------


def persist_restore_cell() -> nbf.NotebookNode:
    return code('''
# === persist: restore results/ from the shared backend =====================
# Each phase runs in a SEPARATE Colab Enterprise runtime, so /content starts
# empty. To see the previous phase's artifacts (discovery.json, the H6 jsonls,
# comparison.csv …) we restore results/ from a shared backend chosen here.
#
# RECOMMENDED on Colab Enterprise: a GCS bucket. Set it once per runtime:
#     %env GCS_BUCKET=gs://your-bucket-name
# (gsutil is pre-installed and the runtime service account has access.)
# Free-Colab fallback: Google Drive is auto-mounted if no bucket is set.
import os, subprocess
from src.persist import detect_backend, build_sync_cmd, remote_location

_drive_ok = Path('/content/drive').exists()
if not os.environ.get('GCS_BUCKET') and not _drive_ok:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        _drive_ok = Path('/content/drive').exists()
    except Exception as _e:
        print('Drive mount unavailable (fine if you are using GCS):', _e)

BACKEND = detect_backend(os.environ, _drive_ok)
BUCKET  = os.environ.get('GCS_BUCKET') or None
REMOTE  = remote_location(BACKEND, bucket=BUCKET) if BACKEND != 'local' else None
print(f'Persistence backend = {BACKEND}   remote = {REMOTE}')

import shutil as _shutil
def _sync(src, dst, backend, label):
    """Run one rsync/gsutil sync, guarding a missing CLI + first-phase noise."""
    cmd = build_sync_cmd(src, dst, backend)
    if _shutil.which(cmd[0]) is None:
        print(f'!! {cmd[0]!r} not on PATH — cannot {label}. '
              f'On Colab Enterprise gsutil is preinstalled; for Drive, rsync is.')
        return
    print(f'{label}:', ' '.join(cmd))
    # capture_output so an empty-remote gsutil CommandException on phase-0
    # restore doesn't dump a scary multi-line stderr; surface it only if it
    # looks like a real failure.
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if r.returncode != 0 and 'does not name a directory' not in (r.stderr or ''):
        tail = (r.stderr or '').strip().splitlines()[-3:]
        if tail:
            print('   (note)', ' | '.join(tail))

RESULTS.mkdir(parents=True, exist_ok=True)
if REMOTE:
    if BACKEND == 'drive':
        Path(REMOTE).mkdir(parents=True, exist_ok=True)
    # remote -> local. check=False semantics: on the FIRST phase the remote is
    # empty, which is not an error.
    _sync(REMOTE, str(RESULTS), BACKEND, 'restore')
    print('Restored results/ from', REMOTE)
else:
    print('!! local backend: this phase will NOT see other phases\\' outputs.')
    print('!! Set GCS_BUCKET (recommended) or mount Drive to chain phases.')
''')


def persist_mirror_cell() -> nbf.NotebookNode:
    return code('''
# === persist: mirror results/ back to the shared backend ===================
# Run this LAST so the next phase's runtime can restore what this phase made.
# (`_sync` was defined in the restore cell — same guards apply.)
if REMOTE:
    if BACKEND == 'drive':
        Path(REMOTE).mkdir(parents=True, exist_ok=True)
    _sync(str(RESULTS), REMOTE, BACKEND, 'mirror')       # local -> remote
    print('Mirrored results/ →', REMOTE)
else:
    print('local backend — results stay in /content/results only this session.')
''')


def discovery_save_cell() -> nbf.NotebookNode:
    return code('''
# === save discovery checkpoint (read by 02/03/04) ==========================
import time
from src.checkpoint import Discovery, save_discovery

# Re-derive the anchor d from the H1 candidate (robust to the `d` global being
# shadowed by H3's drill loop — same guard the monolith H6 cell uses).
_best = next(c for c in cands if c.layer == L_star and c.neuron == N_star)
_anchor_d = float(_best.a_pos - _best.a_neg) or 1e-3

disc = Discovery(
    model_name=MODEL_NAME,
    gate_layer=int(L_star), gate_neuron=int(N_star),
    gate_m_star=float(m_star), gate_anchor_d=float(_anchor_d),
    critical_layer=int(critical),
    layer_scores=[float(mean_per_layer[k]) for k in sorted(mean_per_layer)],
    halluc_neurons=[{'layer': int(n.layer), 'neuron': int(n.neuron)} for n in halluc_neurons[:3]],
    overconf_neurons=[{'layer': int(n.layer), 'neuron': int(n.neuron)} for n in over_neurons[:3]],
    git_sha=sha, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    n_layers=int(lm.n_layers), d_ff=int(lm.d_ff),
    extra={'mean_per_layer': {str(k): float(v) for k, v in mean_per_layer.items()}},
)
save_discovery(RESULTS, disc)
print('Saved', RESULTS / 'discovery.json')
print('  gate        :', disc.gate_dict())
print('  critical    :', disc.critical_layer)
print('  halluc top3 :', disc.halluc_neurons)
print('  overconf t3 :', disc.overconf_neurons)
''')


_RESTORE_COMMON = '''
from src.healthbench import (load_medqa, run_conditions, ablate_neurons_factory,
                             anchor_factory, zero_mlp_factory)
from src.checkpoint import load_discovery

disc = load_discovery(RESULTS)   # raises a clear error if 01_discovery hasn't run
if disc.model_name != MODEL_NAME:
    print(f'!! WARNING: discovery used {disc.model_name} but this runtime loaded '
          f'{MODEL_NAME}.\\n!! Neuron indices are model-specific — re-run '
          f'01_discovery.ipynb on THIS model if the two differ.')
L_star, N_star = disc.gate_layer, disc.gate_neuron
m_star, anchor_d = disc.gate_m_star, disc.gate_anchor_d
critical = disc.critical_layer
top_overconf = disc.overconf_neurons[:3]
top_halluc   = disc.halluc_neurons[:3]
combined     = disc.combined_neurons
N_BENCH = int(os.environ.get('N_BENCH', '1273'))
DATASET = 'GBaker/MedQA-USMLE-4-options-hf'
items = load_medqa(DATASET, split='test', n=N_BENCH, seed=0)
print(f'Restored discovery (gate L{L_star}:F{N_star} m*={m_star} d={anchor_d:.4f}, '
      f'critical L{critical}); loaded {len(items)} MedQA items.')
'''


def h6_setup_from_ckpt_cell() -> nbf.NotebookNode:
    return code('# === H6 — setup conditions from discovery checkpoint ======================='
                + _RESTORE_COMMON + '''
H6_RESULTS = RESULTS / 'h6'; H6_RESULTS.mkdir(exist_ok=True)
_N_GPUS_PRE = torch.cuda.device_count() if torch.cuda.is_available() else 0
H6_MODE = 'DEEP_FULL' if _N_GPUS_PRE >= 4 else 'DEEP'   # 4× GPUs → all 6 conditions

ALL_CONDITIONS = {
    'baseline':           None,
    'h1_gate_anchor':     anchor_factory(lm.layers, L_star, N_star, m_star, anchor_d, k=1.0),
    'h3_zero_layer':      zero_mlp_factory(lm.layers, [critical]),
    'h4_ablate_halluc':   ablate_neurons_factory(lm.layers, top_halluc),
    'h5_ablate_overconf': ablate_neurons_factory(lm.layers, top_overconf),
    'h4_h5_combined':     ablate_neurons_factory(lm.layers, combined),
}
DEEP_KEYS = ['baseline', 'h1_gate_anchor', 'h5_ablate_overconf']
CONDITIONS = ALL_CONDITIONS if H6_MODE in ('FAST', 'DEEP_FULL') else {k: ALL_CONDITIONS[k] for k in DEEP_KEYS}
print(f'Mode = {H6_MODE} → {len(CONDITIONS)} conditions × {len(items)} questions')
''')


def scale_restore_cell() -> nbf.NotebookNode:
    return code('# === restore discovery + rebuild conditions (scale phase) ==================='
                + _RESTORE_COMMON + '''
import types
# H7's layer-overlap cell compares against H5's `over_neurons` (objects with
# .layer/.neuron). Rebuild lightweight shims from the checkpoint.
_NS = lambda d: types.SimpleNamespace(layer=d['layer'], neuron=d['neuron'])
over_neurons = [_NS(d) for d in disc.overconf_neurons]

H4_RESULTS = RESULTS / 'h4'          # H4-extended (if used) reads classifications here
H6_RESULTS = RESULTS / 'h6'; H6_RESULTS.mkdir(exist_ok=True)

# Rebuild the H6 condition factories so H6 pass-2 can resume baseline/h1/h5
# from the restored jsonls and only run the two new H7 conditions.
CONDITIONS = {
    'baseline':           None,
    'h1_gate_anchor':     anchor_factory(lm.layers, L_star, N_star, m_star, anchor_d, k=1.0),
    'h5_ablate_overconf': ablate_neurons_factory(lm.layers, top_overconf),
}
_N_GPUS_PRE = torch.cuda.device_count() if torch.cuda.is_available() else 0
if _N_GPUS_PRE >= 4:   # DEEP_FULL benchmark also wrote these three
    CONDITIONS['h3_zero_layer']    = zero_mlp_factory(lm.layers, [critical])
    CONDITIONS['h4_ablate_halluc'] = ablate_neurons_factory(lm.layers, top_halluc)
    CONDITIONS['h4_h5_combined']   = ablate_neurons_factory(lm.layers, combined)
if not (H6_RESULTS / 'baseline.jsonl').exists():
    print('!! No H6 baseline.jsonl restored — H6 pass-2 will re-run baseline from'
          ' scratch.\\n!! Run 02_benchmark.ipynb first (and mirror to the backend).')
print(f'Rebuilt {len(CONDITIONS)} base conditions for H6 pass-2.')
''')


def h6_timing_probe_cell() -> nbf.NotebookNode:
    return code('''
# === H6 — wall-time probe (times 2 questions before the full run) ==========
# This prints an HONEST estimate so a slow config never silently eats 24 h.
# It runs on the main model (still loaded here, before the free-for-parallel
# cell). With the early-stop fix a question is ~a few seconds, not ~25 s.
import time as _time
from src.healthbench import run_one
_pn = min(2, len(items))
_t = _time.time()
for _it in items[:_pn]:
    _ = run_one(lm, _it, 'baseline')
_per_q = (_time.time() - _t) / max(1, _pn)
_ng = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
_est_h = _per_q * len(items) * len(CONDITIONS) / _ng / 3600
print(f'~{_per_q:.1f}s/question  →  est H6 wall-time {_est_h:.1f} h '
      f'for {len(items)} questions × {len(CONDITIONS)} conditions on {_ng} GPU(s)')
if _est_h > 3:
    print('!! >3 h. For a first pass set a smaller N and re-run THIS notebook:')
    print('!!     %env N_BENCH=300        (then Runtime → Run all)')
    print('!! The run is resumable, so you can scale N_BENCH back up later.')
del _t, _per_q, _est_h
''')


def free_main_model_before_parallel_cell() -> nbf.NotebookNode:
    return code('''
# === free the main model before the multi-GPU H6 run =======================
# On 4× A100-40 the data-parallel path spawns one worker per GPU, and EACH
# worker loads its own NF4 copy (~18 GB for Qwen3-32B). The main-process copy
# still sits on GPU0, so worker-0 would try to fit a second ~18 GB model on
# the same 40 GB card → OOM on long reasoning chains. Since the parallel path
# rebuilds every intervention from JSON specs inside the workers, the main
# copy is dead weight here — drop it (and the lm-bound CONDITIONS closures,
# which otherwise keep the weights alive) so each worker owns a clean GPU.
#
# NB: only the *parallel* branch is freed. On a single GPU we keep the model
# because run_conditions() runs in THIS process. The scalar neuron coords
# (L_star, top_overconf, …) survive either way — the run cell rebuilds the
# specs from them.
import gc, torch
_N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
if _N_GPUS > 1:
    # Both dicts hold factory closures over lm.layers → they pin the weights.
    for _cname in ('CONDITIONS', 'ALL_CONDITIONS'):
        try:
            del globals()[_cname]
        except KeyError:
            pass
    try:
        if 'lm' in dir() and lm is not None:
            for _attr in ('model', 'tokenizer', 'layers'):
                if hasattr(lm, _attr):
                    setattr(lm, _attr, None)
        lm = None
    except Exception as _e:
        print('main-model free skipped:', _e)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'Freed main model before parallel H6 ({_N_GPUS} GPUs → workers own them).')
else:
    print('Single GPU — keeping the main model for the sequential H6 path.')
''')


def periodic_mirror_start_cell() -> nbf.NotebookNode:
    return code('''
# === start a periodic mirror for the duration of the H6 run ================
# The benchmark writes per-worker shards to results/h6/_gpu{rank}/ and only
# the END-of-phase mirror would otherwise push them. A disconnect mid-run
# would then lose all partial shards → the next runtime restarts from zero.
# Mirroring every 3 min bounds the loss; restore is recursive so the partial
# _gpu* shards come back and each worker resumes positionally from its shard.
_mirror = None
if REMOTE:
    from src.persist import PeriodicMirror
    _mirror = PeriodicMirror(str(RESULTS), REMOTE, BACKEND, interval=180).start()
    print(f'Periodic mirror running every 180s → {REMOTE}')
else:
    print('local backend — no periodic mirror (results stay in this runtime).')
''')


def periodic_mirror_stop_cell() -> nbf.NotebookNode:
    return code('''
# === stop the periodic mirror (does one final sync) ========================
if _mirror is not None:
    _mirror.stop(final=True)
    print(f'Periodic mirror stopped after {_mirror.n_syncs} syncs (+ final).')
''')


def sycophancy_restore_cell() -> nbf.NotebookNode:
    return code('''
# === restore items for the sycophancy phase ================================
from src.healthbench import load_medqa
H6_RESULTS = RESULTS / 'h6'          # hardest-case selection reads comparison.csv if present
N_BENCH = int(os.environ.get('N_BENCH', '1273'))
items = load_medqa('GBaker/MedQA-USMLE-4-options-hf', split='test', n=N_BENCH, seed=0)
_has_comp = (H6_RESULTS / 'comparison.csv').exists()
print(f'Loaded {len(items)} MedQA items. H6 comparison.csv present: {_has_comp}')
if not _has_comp:
    print('  (No H6 comparison.csv — sycophancy will fall back to a random subsample'
          ' instead of the baseline-wrong hardest cases.)')
''')


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

NB_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "machine_shape": "hm"},
    "accelerator": "GPU",
}


def _title(phase_no: str, name: str, blurb: str) -> nbf.NotebookNode:
    """Banner as a CODE cell (no markdown cells in these notebooks — the user
    asked for code-only; orienting text lives in comments + a print)."""
    blurb_lines = "\n".join("# " + ln for ln in _wrap_comment(blurb))
    return code(
        "# ==========================================================================\n"
        f"# Diagnostic Percept — {phase_no} · {name}\n"
        f"{blurb_lines}\n"
        "# Run phases in order 00 -> 01 -> 02 -> 03 -> 04, each a separate Colab\n"
        "# Enterprise runtime. State is shared through results/ mirrored to a GCS\n"
        "# bucket (set GCS_BUCKET) or Drive. Qwen3 only (32B/14B/8B/4B by GPU mem).\n"
        "# ==========================================================================\n"
        f"print('=== Diagnostic Percept | {phase_no} {name} ===')"
    )


def _wrap_comment(text: str, width: int = 74):
    import textwrap
    return textwrap.wrap(text, width=width) or [""]


def write_nb(fname: str, cells: List[nbf.NotebookNode]) -> Path:
    # Code-only notebooks: drop every markdown cell. Section context is already
    # carried by the `# === label ===` comment at the top of each code cell, so
    # nothing executable is lost.
    code_cells = [c for c in cells if c.cell_type == "code"]
    nb = nbf.v4.new_notebook()
    nb.metadata = NB_META
    nb.cells = code_cells
    out = OUT_DIR / fname
    nbf.write(nb, out)
    return out


def build_00_setup() -> List[nbf.NotebookNode]:
    cells = [_title("00", "Setup & health check",
                    "Install deps, redirect caches off the small boot disk, "
                    "pick the Qwen3 model for this GPU, load it, and verify the "
                    "MLP hooks + gradient flow. **Run this first** to confirm a "
                    "fresh runtime is healthy before launching a multi-hour job.")]
    cells += setup_cells()
    cells.append(code('''
# === readiness summary =====================================================
import torch
print('=' * 60)
print('Model        :', MODEL_NAME)
print('Layers / d_ff:', lm.n_layers, '/', lm.d_ff)
print('dtype/device :', lm.dtype, '/', lm.device)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f'GPU{i} {torch.cuda.get_device_name(i):<18} '
              f'free {free/1e9:5.1f} / {total/1e9:5.1f} GB')
import shutil
for p in ('/', '/content'):
    if Path(p).exists():
        s = shutil.disk_usage(p); print(f'disk {p:<9} free {(s.total-s.used)/1e9:6.1f} GB')
print('=' * 60)
print('READY. Proceed to 01_discovery.ipynb (same GCS_BUCKET / Drive).')
'''))
    return cells


def build_01_discovery() -> List[nbf.NotebookNode]:
    cells = [_title("01", "Discovery (H1–H5)",
                    "Find the diagnosis-gate neuron (H1), disease concept "
                    "neurons (H2), the symptom→diagnosis routing layer (H3), "
                    "hallucination neurons (H4 + per-category), and "
                    "overconfidence neurons (H5). Cheap (~20–30 min on one "
                    "A100). Writes `results/discovery.json` for later phases.")]
    cells += setup_cells()
    cells.append(persist_restore_cell())
    # H1
    cells.append(md_containing("## 3. H1 — diagnosis-gate neuron"))
    for lab in ("H1 — load data, discover",
                "H1 — multiplier sweep over top-5 + capability cost",
                "H1 — activation distribution at winning neuron",
                "H1 — hard-case capability check (anchor Eq. 7)",
                "free VRAM after H1"):
        cells.append(code_by_label(lab))
    # H2
    cells.append(md_containing("## 4. H2 — disease-specific concept neurons"))
    for lab in ("H2 — build corpus, rank concept neurons",
                "H2 — amplification matrix (relative multipliers)"):
        cells.append(code_by_label(lab))
    # H3
    cells.append(md_containing("## 5. H3 — symptom→diagnosis routing"))
    for lab in ("H3 — verify diagnosis tokens",
                "H3 — per-layer patching curve",
                "H3 — full d_ff drill at critical layer",
                "free VRAM after H3 drill"):
        cells.append(code_by_label(lab))
    # H4 + extended
    cells.append(md_containing("## 6. H4 — hallucination"))
    for lab in ("H4 — classify + find hallucination neurons",
                "H4 — layer profile of hallucination signal",
                "free VRAM after H4"):
        cells.append(code_by_label(lab))
    cells.append(md_containing("## 11. H4-extended"))
    cells.append(code_by_label("H4 — per-category commit rates from cached classifications"))
    # H5
    cells.append(md_containing("## 7. H5 — overconfidence"))
    for lab in ("H5 — measure calibration on hard cases + rank neurons",
                "H5 — plot calibration scatter + neuron correlation"):
        cells.append(code_by_label(lab))
    # checkpoint + mirror
    cells.append(md("## Save discovery checkpoint + mirror results"))
    cells.append(discovery_save_cell())
    cells.append(persist_mirror_cell())
    return cells


def build_02_benchmark() -> List[nbf.NotebookNode]:
    cells = [_title("02", "Benchmark (H6 + consensus-flip)",
                    "Run the MedQA-USMLE test set (1273 questions) under all "
                    "intervention conditions, multi-GPU data-parallel on 4× "
                    "A100, then the consensus-flip enrichment analysis. The "
                    "expensive phase (~2 h on 4× A100). Fully resumable: re-run "
                    "after a disconnect and it picks up from the jsonls.")]
    cells += setup_cells()
    cells.append(persist_restore_cell())
    cells.append(md_containing("## 8. H6 — Benchmark eval under interventions"))
    cells.append(h6_setup_from_ckpt_cell())
    cells.append(h6_timing_probe_cell())
    cells.append(free_main_model_before_parallel_cell())
    cells.append(periodic_mirror_start_cell())
    cells.append(code_by_label("H6 — run all conditions (resumable, multi-GPU if available)"))
    cells.append(periodic_mirror_stop_cell())
    cells.append(code_by_label("H6 — sample reasoning per condition"))
    cells.append(code_by_label("H6 — comparison table + delta plot"))
    cells.append(md_containing("## 9. Consensus-flip analysis"))
    cells.append(code_by_label("consensus-flip analyzer"))
    cells.append(md("## Mirror results"))
    cells.append(persist_mirror_cell())
    return cells


def build_03_scale() -> List[nbf.NotebookNode]:
    cells = [_title("03", "Scale analyses (H7 · H6 pass-2 · H8 · MedMCQA)",
                    "MedQA-scale calibration-failure layers (H7), the "
                    "H7-informed causal re-test (H6 pass-2), the cross-task "
                    "confidence-circuit split (H8), and the MedMCQA "
                    "replication. Needs the H6 outputs from phase 02.")]
    cells += setup_cells()
    cells.append(persist_restore_cell())
    cells.append(scale_restore_cell())
    # H7
    cells.append(md_containing("## 10. H7 — Calibration-failure layers"))
    cells.append(code_by_label("H7 — collect answer-position activations + rank miscalibration neurons"))
    cells.append(code_by_label("H7 — layer profile + comparison to H5"))
    # H6 pass-2
    cells.append(md_containing("## 10b. H6 pass-2"))
    cells.append(code_by_label("H6 pass-2 — add H7 conditions and run incrementally"))
    cells.append(code_by_label("free VRAM after H7"))
    # H8
    cells.append(md_containing("## 10c. H8 — Cross-task confidence circuits"))
    cells.append(code_by_label("H8 — collect MCQ + prose activations on the same questions"))
    cells.append(code_by_label("H8 — scatter of r_mcq vs r_prose + layer profile"))
    # MedMCQA
    cells.append(md_containing("## 11b. MedMCQA replication"))
    cells.append(code_by_label("MedMCQA — load + run + consensus-flip"))
    cells.append(md("## Mirror results"))
    cells.append(persist_mirror_cell())
    return cells


def build_04_sycophancy() -> List[nbf.NotebookNode]:
    cells = [_title("04", "Sycophancy probe + reduction",
                    "Three forwards per item (baseline / authority push / "
                    "insistence push) to measure when the model abandons its "
                    "answer to match a wrong user claim, then the contrastive "
                    "gradient×activation pass that isolates the sycophancy "
                    "circuit, and an ablation reduction test.")]
    cells += setup_cells()
    cells.append(persist_restore_cell())
    cells.append(sycophancy_restore_cell())
    cells.append(md_containing("## 11c. Sycophancy"))
    for lab in ("sycophancy — probe a hardest-case subset",
                "sycophancy — find neurons + layer rise curve",
                "sycophancy — reduction: ablate top neurons + re-probe"):
        cells.append(code_by_label(lab))
    cells.append(md("## Mirror results"))
    cells.append(persist_mirror_cell())
    return cells


def main() -> None:
    built = [
        write_nb("00_setup_check.ipynb", build_00_setup()),
        write_nb("01_discovery.ipynb", build_01_discovery()),
        write_nb("02_benchmark.ipynb", build_02_benchmark()),
        write_nb("03_scale.ipynb", build_03_scale()),
        write_nb("04_sycophancy.ipynb", build_04_sycophancy()),
    ]
    for p in built:
        nb = nbf.read(p, as_version=4)
        print(f"wrote {p.relative_to(ROOT)}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
