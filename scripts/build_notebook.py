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
import os, sys, subprocess, json, time, traceback
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
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR], check=True)
else:
    print('Pulling', REPO_DIR, '...')
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=False)
sys.path.insert(0, str(Path(REPO_DIR).resolve()))

RESULTS = Path('/content/results'); RESULTS.mkdir(parents=True, exist_ok=True)
print('Repo   :', Path(REPO_DIR).resolve())
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

cells.append(code(wrap("H1 — multiplier sweep over top-5", """
from types import SimpleNamespace
sweep_path = H1_RESULTS / 'sweep.json'
probe = h1['positive'][:8]

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

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
by_neuron = {}
for s in sweep_raw:
    by_neuron.setdefault((s['layer'], s['neuron']), []).append((s['multiplier'], s['target_logprob']))
for (L, N), pts in by_neuron.items():
    pts.sort()
    xs, ys = zip(*pts)
    ax.plot(xs, ys, marker='o', label=f'L{L}:F{N}')
ax.set_xlabel('multiplier m'); ax.set_ylabel('mean target log-prob (lower = suppressed)')
ax.set_title('H1: top-5 suppression sweep'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(H1_RESULTS / 'sweep.png', dpi=140); plt.show()

L_star, N_star, m_star = best_multiplier([SimpleNamespace(**s) for s in sweep_raw])
print(f'\\nBest gate: L{L_star}:F{N_star} at m={m_star}')
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

cells.append(code(wrap("H1 — MedQA-mini capability check (Eq. 7 anchor)", """
from src.hooks import anchor_intervention
from src.eval import score_hedging

MEDQA_MINI = [
    'What is the first-line treatment for hypertension?',
    'Name the four chambers of the heart.',
    'What is the normal range for fasting glucose?',
    'What does HbA1c measure?',
    'What is the mechanism of beta-blockers?',
    'Name three causes of microcytic anemia.',
    'What is the difference between Type 1 and Type 2 diabetes?',
    'What are the components of CHA2DS2-VASc?',
    'What is the antidote for acetaminophen overdose?',
    'What is the Glasgow Coma Scale?',
    'Define sepsis.',
    'What are red flag symptoms in low back pain?',
    'What is the mechanism of metformin?',
    'Name three side effects of corticosteroids.',
    'What is the function of the liver?',
    'What are the stages of chronic kidney disease?',
    'What is the most common cause of community-acquired pneumonia?',
    'What are the criteria for diagnosing major depression?',
    'What is the function of the parathyroid hormone?',
    'Define anaphylaxis.',
]

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
for q in MEDQA_MINI:
    baseline = gen(q, 48)
    with anchor_intervention(lm.layers, N_star, m_star, d, L_star, k=1.0):
        anchor = gen(q, 48)
    capability.append({'q': q, 'baseline': baseline, 'anchor': anchor})

for row in capability[:3]:
    print('Q :', row['q'])
    print(' baseline:', row['baseline'][:200])
    print(' anchor  :', row['anchor'][:200])
    print()
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

cells.append(code(wrap("H2 — amplification matrix (disease × prompt × multiplier)", """
amp_path = H2_RESULTS / 'amplification.json'
benign = h2['_benign_prompts']['positive'][:4]
multipliers = [0.0, 20.0, 80.0, 160.0]

if amp_path.exists():
    amp_results = json.loads(amp_path.read_text())
    print('Reloaded amplification.')
else:
    amp_results = {}
    for disease, neurons in concepts.items():
        c = neurons[0]
        print(f'Amplifying {disease} via L{c.layer}:F{c.neuron} ...')
        rows = amplification_matrix(
            lm, neuron=c, benign_prompts=benign, multipliers=multipliers,
            max_new_tokens=64, concept_keywords=DISEASE_KEYWORDS[disease],
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

cells.append(code(wrap("H3 — neuron drill at critical layer", """
mean_per_layer = {}
for L in range(lm.n_layers):
    vals = [r['score'] for rows in patch_data.values() for r in rows if r['layer'] == L]
    if vals:
        mean_per_layer[L] = float(np.mean(vals))
critical = max(mean_per_layer, key=mean_per_layer.get)
print(f'Critical layer (mean score {mean_per_layer[critical]:+.3f}): L{critical}')

drill_path = H3_RESULTS / f'drill_L{critical}.json'
if drill_path.exists():
    drill_data = json.loads(drill_path.read_text())
    print('Reloaded drill data.')
else:
    pair = H3_PAIRS[0]
    stride = max(1, lm.d_ff // 256)
    drill = patch_neurons_at_layer(
        lm, pair.clean_prompt, pair.corrupted_prompt,
        pair.clean_dx, pair.corrupted_dx, layer_idx=critical,
        neuron_indices=list(range(0, lm.d_ff, stride)),
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

cells.append(md("## 6. Persist run metadata"))

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
