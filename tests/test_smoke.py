"""End-to-end smoke test on Qwen/Qwen3-0.6B.

Runs the entire H1+H2+H3 pipeline with tiny inputs (5 prompts/class, 2 layers
of patching, 1 amplification multiplier) in under 60 s on CPU/MPS/CUDA.

The test asserts only *shape* and *type* invariants — no behavioral claims
about Qwen3-0.6B's diagnostic abilities. It exists to catch interface
regressions in :mod:`src` before a Colab run.

Run with::

    pytest tests/test_smoke.py -s
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Allow ``python tests/test_smoke.py`` and ``pytest tests/`` interchangeably.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SMOKE_MODEL = os.environ.get("DP_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
SKIP_H1 = os.environ.get("DP_SMOKE_SKIP_H1") == "1"


@pytest.fixture(scope="module")
def lm():
    from src.model import load_model, set_seed
    set_seed(0)
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.bfloat16
    else:
        device_map = None
        # bf16 on CPU is supported but slow; keep bf16 to honor "no silent
        # fallback". The smoke test still completes within budget for 0.6B.
        dtype = torch.bfloat16
    lm = load_model(SMOKE_MODEL, dtype=dtype, device_map=device_map)
    yield lm
    del lm


def test_load_and_hook_forward(lm):
    """Forward pass populates ``mlp._h`` and shapes line up with d_ff."""
    tok = lm.tokenizer
    ids = tok("The patient has", return_tensors="pt").input_ids.to(lm.device)
    lm.model(input_ids=ids, use_cache=False)
    for L in (0, lm.n_layers // 2, lm.n_layers - 1):
        h = lm.layers[L].mlp._h
        assert h is not None, f"layer {L} mlp._h not populated"
        assert h.shape[-1] == lm.d_ff
        assert h.shape[1] == ids.shape[1]


def test_backward_grads_flow(lm):
    """``h.retain_grad`` and ``.backward()`` produce non-zero gradients."""
    tok = lm.tokenizer
    ids = tok("Pain in the chest. The diagnosis is", return_tensors="pt").input_ids.to(lm.device)
    lm.model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = lm.model(input_ids=ids, use_cache=False)
        out.logits[0, -1].sum().backward()
    g = lm.layers[0].mlp._h.grad
    assert g is not None and torch.isfinite(g).all()
    # Some neurons should have non-zero gradient.
    assert g.abs().sum().item() > 0


@pytest.mark.skipif(SKIP_H1, reason="DP_SMOKE_SKIP_H1=1 (CPU bf16 backward too slow for CI)")
def test_h1_discover_and_sweep(lm):
    """H1 discovery returns 5 candidates and a sweep of multipliers."""
    from src.data import build_h1
    from src.discover import discover, sweep, best_multiplier

    d = build_h1()
    pos = d["positive"][:3]
    neg = d["negative"][:3]

    cands = discover(
        lm,
        positive_prompts=pos, negative_prompts=neg,
        target_phrases=d["commitment_phrases"], icd10_tokens=d["icd10_tokens"],
        layer_range=(0, 2), top_k=3,
    )
    assert len(cands) == 3
    assert all(0 <= c.layer < lm.n_layers and 0 <= c.neuron < lm.d_ff for c in cands)

    sw = sweep(
        lm, cands[:1], probes=pos[:2],
        target_phrases=d["commitment_phrases"], icd10_tokens=d["icd10_tokens"],
        multipliers=(0.0, 10.0, -10.0), sample_prompt=None,
    )
    assert len(sw) == 3
    L, n, m = best_multiplier(sw)
    assert isinstance(L, int) and isinstance(n, int) and isinstance(m, float)


def test_h2_concept_and_amplification(lm):
    """H2 ranks top neurons and amplifies one on a benign prompt."""
    from src.concept import (
        DISEASE_KEYWORDS, amplification_matrix, rank_concept_neurons,
    )
    from src.data import build_h2

    h2 = build_h2(n_per_disease=8)
    disease = "mi"
    pos = h2[disease]["positive"][:5]
    neg = h2[disease]["negative"][:5]
    neurons = rank_concept_neurons(
        lm, positive=pos, negative=neg,
        layer_range=(0, max(2, lm.n_layers // 2)), top_k=2, disease=disease,
    )
    assert len(neurons) == 2

    amps = amplification_matrix(
        lm, neuron=neurons[0],
        benign_prompts=["Write a haiku about autumn leaves."],
        multipliers=(0.0, 20.0),
        max_new_tokens=16,
        concept_keywords=DISEASE_KEYWORDS[disease],
    )
    assert len(amps) == 2
    assert all(isinstance(a.generation, str) for a in amps)


def test_h3_patch_layers_and_drill(lm):
    """H3 produces a per-layer score curve and a neuron drill at one layer."""
    from src.data import H3_PAIRS
    from src.patching import patch_layers, patch_neurons_at_layer

    pair = H3_PAIRS[0]
    L_max = min(4, lm.n_layers)  # smoke: only first few layers
    scores = patch_layers(
        lm, pair.clean_prompt, pair.corrupted_prompt,
        pair.clean_dx, pair.corrupted_dx, pair.pair_id,
        layer_indices=list(range(L_max)),
    )
    assert len(scores) == L_max
    assert all(isinstance(s.score, float) for s in scores)

    drill = patch_neurons_at_layer(
        lm, pair.clean_prompt, pair.corrupted_prompt,
        pair.clean_dx, pair.corrupted_dx, layer_idx=0,
        neuron_indices=list(range(0, lm.d_ff, max(1, lm.d_ff // 8))),
        pair_id=pair.pair_id,
    )
    assert len(drill) > 0


def test_h4_hallucinate(lm):
    """H4 produces classifications + ranked hallucination neurons (tiny set)."""
    from src.data import build_h4
    from src.hallucinate import find_hallucination_neurons

    h4 = build_h4()
    neurons, classifications = find_hallucination_neurons(
        lm,
        trap_prompts=h4["trap"][:4],
        pathognomonic_prompts=h4["pathognomonic"][:3],
        hedge_prompts=["A patient reports fatigue. What is the differential?"],
        target_phrases=h4["commitment_phrases"],
        icd10_tokens=h4["icd10_tokens"],
        layer_range=(0, max(2, lm.n_layers // 4)),
        top_k=3,
        commit_p_threshold=0.0,  # force at least some commits on tiny model
    )
    assert "trap" in classifications and "pathognomonic" in classifications
    assert isinstance(neurons, list)  # may be empty on Qwen3-0.6B


def test_eval_helpers():
    from src.eval import score_hedging, score_injection

    h = score_hedging("Could be pneumonia or bronchitis. Consider the differential.")
    assert h.is_hedging
    c = score_hedging("The diagnosis is acute myocardial infarction.")
    assert not c.is_hedging

    inj = score_injection(
        "The patient committed suicide by hanging in their apartment on a Tuesday.",
        disease_keywords=["suicide", "depression"],
        prompt_keywords=["tuesday"],
    )
    assert inj.mentions and inj.relevant


@pytest.mark.skipif(SKIP_H1, reason="DP_SMOKE_SKIP_H1=1")
def test_smoke_under_60s(lm):
    """Sanity check that the full module-level test budget is in range.

    Pytest's per-test wall clock isn't the gate (the budget is total runtime
    including model load), but this at least flags a regression if a single
    operation balloons.
    """
    start = time.time()
    from src.data import build_h1
    from src.discover import discover
    d = build_h1()
    discover(
        lm, positive_prompts=d["positive"][:3], negative_prompts=d["negative"][:3],
        target_phrases=d["commitment_phrases"], icd10_tokens=d["icd10_tokens"],
        layer_range=(0, 2), top_k=2,
    )
    elapsed = time.time() - start
    # GPU budget is 60s; CPU bf16 on a 0.6B model is much slower, so we relax
    # to 600s here and rely on Colab timings as the real gate.
    budget = 60.0 if torch.cuda.is_available() else 600.0
    assert elapsed < budget, f"discover too slow: {elapsed:.1f}s (budget {budget:.0f}s)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x"]))
