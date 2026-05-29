# Diagnostic Percept

[![CI](https://github.com/ArioMoniri/diagnosticpercept/actions/workflows/ci.yml/badge.svg)](https://github.com/ArioMoniri/diagnosticpercept/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArioMoniri/diagnosticpercept/blob/main/notebooks/diagnostic_percept.ipynb)

Porting Kazemi et al. 2026 ("A Single Neuron Is Sufficient to Bypass Safety
Alignment in LLMs", [arXiv 2605.08513](https://arxiv.org/abs/2605.08513)) from
safety alignment to **clinical diagnosis** in a medical LLM.

## Run it

1. **Open the notebook in Colab** via the badge above
2. (Optional) Add `GH_TOKEN` as a Colab secret if you want the git-bus bridge to push results back
3. Runtime → A100 GPU, High-RAM
4. Run all cells — automation covers model load → H1 → H8 → results JSON

The default model chain is **Qwen-only** (Qwen3 size ladder: 32B → 14B → 8B → 4B). Qwen3.5/3.6 checkpoints don't load cleanly on standard transformers and are skipped.
No HF token required — all are open weights. Override with
`os.environ['MODEL_OVERRIDE']='Qwen/<repo-name>'` before the load cell.

**For long unattended runs without disconnects, switch to Colab Enterprise**
on Google Cloud — see [`docs/colab-enterprise.md`](docs/colab-enterprise.md)
for the runtime template config and a `scripts/submit_to_vertex.py` batch
runner.

## Hypotheses

| ID | Claim | Method |
|----|-------|--------|
| **H1** | A single MLP neuron acts as a *diagnosis gate*: suppressing it flips committed diagnosis ("The diagnosis is X") to hedging ("Could be X, Y, or Z"). | Paper §2.3 — gradient × activation on a contrastive set (pathognomonic vs. ambiguous vignettes), top-5 reranking by multiplier sweep, constant (Eq. 5) + anchor (Eq. 7) interventions. |
| **H2** | Each disease in {sepsis, T2DM, acute MI, pneumonia, asthma, major depression} has *concept neurons* whose mean activation on disease-positive sentences exceeds disease-negative by a large standardized margin; amplifying them on benign prompts induces disease injection. | Paper §4 — corpus-based max-activation ranking, additive amplification (`h_i ← h_i + m`). |
| **H3** | Symptom-to-diagnosis routing flows through identifiable layers, visible via residual-stream activation patching across clean (e.g. chest-pain → MI) / corrupted (dyspnea → asthma) vignette pairs. | Meng et al. 2022 ROME / Wang et al. 2022 IOI patching; per-layer score = `(patched − corrupted) / (clean − corrupted)`, then drill into the critical layer at the MLP-neuron level. |

## Method recap (paper §2.3, summarized)

- Hook the **pre-down-projection** MLP activation
  `h = SiLU(W_gate x) ⊙ (W_up x) ∈ R^{d_ff}` (Eq. before §2.3).
- Loss: log-odds of a target-phrase token set
  `L = -log p_target / (1 - p_target)` (**Eq. 2**).
- Combined gradient `G = g^(pos) + g^(neg)` (**Eq. 3**).
- Per-token score `score_{i,t} = G_{i,t} · (a^(neg) − a^(pos))` (**Eq. 4**).
- Magnitude filter: keep only neurons with `|a^(pos)| > |a^(neg)|`.
- Constant intervention `h_i ← m` (**Eq. 5**).
- Anchor intervention `h_i ← clamp(k · m* · v/d, m*)` with `v` from **Eq. 6** and
  `d = a^(pos) − a^(neg)` at the discovery token (**Eq. 7**).

## Repo layout

```
diagnosticpercept/
├── README.md
├── requirements.txt
├── notebooks/diagnostic_percept.ipynb   # main Colab notebook (self-contained, clones repo at top)
├── src/
│   ├── model.py        # load + LlamaMLP / Qwen2MLP forward patching (exposes h.retain_grad)
│   ├── hooks.py        # constant_intervention, additive_intervention, residual_patch ctx managers
│   ├── discover.py     # H1: gradient × activation, top-k, multiplier sweep
│   ├── concept.py      # H2: per-disease corpus, max-activation ranking, amplification eval
│   ├── patching.py     # H3: residual-stream patching, neuron-level drill
│   ├── eval.py         # hedge detector, disease-injection judge, logit-diff
│   └── data.py         # builds data/*.json contrastive sets + disease corpora
├── data/                                 # JSON artifacts produced by src/data.py
├── results/                              # gitignored
└── tests/test_smoke.py                   # full pipeline on Qwen3-0.6B in <60s
```

## Setup

```bash
pip install -r requirements.txt
python -m src.data        # writes data/*.json
pytest tests/test_smoke.py -s
```

## Models

Default candidate chain (first one that exists on HF wins):

- Qwen3 family (`Qwen/Qwen3-32B`, `Qwen3-14B`, `Qwen3-8B`, `Qwen3-4B`)
- Smoke-test tiny: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)

All are open-weights; no HF token required. NF4 4-bit quantization kicks in
automatically below 24 GB VRAM. Hardware target: Colab Pro A100 or larger.

## Running

Open `notebooks/diagnostic_percept.ipynb` in Colab — it clones this repo at the top
and imports from `src/`. Each major section persists artifacts to `results/` and
can be re-run independently after Section 2.

## Reproducibility

- `SEED=0` set in every entrypoint
- bf16 throughout (no silent fp16/fp32 fallback)
- Results JSON includes model name, git SHA, timestamp, top neurons, multipliers,
  judge outputs, plot paths

## Citation

```
@article{kazemi2026singleneuron,
  title={A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models},
  author={Kazemi, Hamid and Chegini, Atoosa and Safi, Maria},
  journal={arXiv preprint arXiv:2605.08513},
  year={2026}
}
```
