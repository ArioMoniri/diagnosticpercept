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
    "!pip -q install 'transformers>=4.44' accelerate bitsandbytes scikit-learn matplotlib tqdm datasets nbformat 2>&1 | tail -5"
))

# Set CUDA alloc config BEFORE torch imports anywhere — must be very first.
cells.append(code("""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
"""))

cells.append(code(wrap("env check", f"""
import os, sys, subprocess, json, time, traceback, importlib
from pathlib import Path
import torch

print('Python:', sys.version.split()[0])
print('Torch :', torch.__version__)
print('CUDA  :', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')
if torch.cuda.is_available():
    print('Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')

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

cells.append(code(wrap("HF login (gated Med42)", """
import os
if not os.environ.get('HF_TOKEN'):
    try:
        from google.colab import userdata
        os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
        print('HF_TOKEN loaded from Colab secrets.')
    except Exception as e:
        print('Colab secret not available:', e)
        from huggingface_hub import notebook_login
        notebook_login()
else:
    print('HF_TOKEN already set.')
""")))

cells.append(md(
    "## 2. Load Med42-8B and verify hooks\n\n"
    "Patches every MLP forward to expose `h = SiLU(W_gate x) * (W_up x)` with "
    "`retain_grad`. Falls back to OpenBioLLM-8B if Med42's license is not yet "
    "accepted on your HF account."
))

cells.append(code(wrap("load model", """
from src.model import load_model, set_seed
set_seed(0)

PRIMARY = 'm42-health/Llama3-Med42-8B'
FALLBACK = 'aaditya/Llama3-OpenBioLLM-8B'

# Auto-pick precision: 8B in bf16 = ~16 GB weights. Need ~24 GB total VRAM
# for safe fwd+bwd. Below that, use 4-bit NF4 (~5 GB weights, bf16 compute).
if torch.cuda.is_available():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    total_gb = 0
USE_4BIT = total_gb < 24.0
print(f'GPU total: {total_gb:.1f} GB  |  use_4bit = {USE_4BIT}')

token = os.environ.get('HF_TOKEN')
try:
    print(f'Trying {PRIMARY} ...')
    lm = load_model(PRIMARY, token=token, quantize_4bit=USE_4BIT)
    MODEL_NAME = PRIMARY
except Exception as e:
    print(f'Med42 load failed: {e}')
    print(f'Falling back to {FALLBACK} ...')
    lm = load_model(FALLBACK, token=token, quantize_4bit=USE_4BIT)
    MODEL_NAME = FALLBACK

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    used = torch.cuda.memory_allocated() / 1e9
    print(f'\\nLoaded: {MODEL_NAME}')
    print(f'  layers = {lm.n_layers}')
    print(f'  d_ff   = {lm.d_ff}')
    print(f'  dtype  = {lm.dtype}')
    print(f'  device = {lm.device}')
    print(f'  VRAM used after load: {used:.2f} GB')
else:
    print(f'\\nLoaded: {MODEL_NAME} (CPU)')
    print(f'  layers = {lm.n_layers}  d_ff = {lm.d_ff}')
""")))

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
# For Med42-8B (d_ff=14336) that's ~14k forwards through one pair on the
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

# H6 mode selection:
#   FAST     = 100 q × all 6 conditions  → ~30 min  (broad scan)
#   DEEP     = 1273 q × 3 conditions     → ~3 hr    (full MedQA on the 3 informative conditions)
# Override by setting N_BENCH / CONDITION_KEYS below.
H6_MODE = 'DEEP'  # 'FAST' or 'DEEP'
N_BENCH = 1273 if H6_MODE == 'DEEP' else 100
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

cells.append(code(wrap("H6 — run all conditions (resumable)", """
# Resumable: if `h6/<condition>.jsonl` exists, picks up from the next item.
# Wall time per condition × question depends on reasoning chain length;
# expect 2-4 s/q on Blackwell. DEEP mode (1273 × 3) ≈ 2-3 hr,
# FAST mode (100 × 6) ≈ 30 min.
import time
t0 = time.time()
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
        print(f'\\n[{verdict}] gold={r.gold}  pred={r.predicted}  p_top1={r.p_top1:.3f}')
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

cells.append(md("## 12. Persist run metadata"))

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
