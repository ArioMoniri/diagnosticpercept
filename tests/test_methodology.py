"""Iteration-3 methodology regression tests (M4, M6, M7, M8).

These run in CPU-only seconds with no model load — they exercise the
non-model-side methodology fixes from the 2026-05-29 ml-developer review:

  * M4 — H6 prompt ensemble: five paraphrases, each ends with `Answer:`.
  * M6 — consensus.implied_letter: stricter commitment regex + fallback.
  * M7 — H7 length-binned ranker: handles small N + returns list.
  * M8 — H1 contrastive set: POSITIVE/NEGATIVE word counts within 3 words.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# M4 — H6 prompt ensemble
# ---------------------------------------------------------------------------


def test_m4_prompt_ensemble_has_five_templates():
    from src.healthbench import list_prompt_templates
    templates = list_prompt_templates()
    assert len(templates) == 5
    # Distinct wording (no copy-paste duplicates).
    assert len(set(templates)) == 5


def test_m4_each_template_renders_with_answer_trailer():
    from src.healthbench import MCQItem, list_prompt_templates, render_prompt
    item = MCQItem(
        q_id="t", question="Q?", options={"A": "x", "B": "y", "C": "z", "D": "w"},
        gold="A",
    )
    for tpl in list_prompt_templates():
        p = render_prompt(item, template=tpl)
        assert "Answer:" in p
        assert "A. x" in p and "D. w" in p
        assert "Q?" in p


# ---------------------------------------------------------------------------
# M6 — stricter consensus regex
# ---------------------------------------------------------------------------


def test_m6_commitment_phrase_overrides_frequency():
    """Many 'A' mentions but one 'the answer is D' → D wins."""
    from src.consensus import implied_letter
    text = "Option A is unlikely. A could fit. A again. The answer is D."
    assert implied_letter(text, ["A", "B", "C", "D"]) == "D"


def test_m6_last_commitment_wins():
    from src.consensus import implied_letter
    text = "The answer is A initially. On reflection, the answer is C."
    assert implied_letter(text, ["A", "B", "C", "D"]) == "C"


def test_m6_falls_back_to_frequency_without_commitment():
    from src.consensus import implied_letter
    text = "C looks right. C fits. C aligns. Not B. Not A."
    assert implied_letter(text, ["A", "B", "C", "D"]) == "C"


def test_m6_tie_returns_none():
    from src.consensus import implied_letter
    text = "A is possible. B is possible."
    assert implied_letter(text, ["A", "B", "C", "D"]) is None


def test_m6_i_choose_pattern():
    from src.consensus import implied_letter
    text = "Various options exist. I choose B because of the findings."
    assert implied_letter(text, ["A", "B", "C", "D"]) == "B"


# ---------------------------------------------------------------------------
# M7 — H7 length-binned ranker
# ---------------------------------------------------------------------------


def test_m7_length_binned_smoke():
    """Length-binned ranker accepts rows with chain_len + returns a list."""
    import random

    from src.h7_layers import rank_miscalibration_neurons

    random.seed(0)
    rows = [
        {
            "q_id": f"q{i}", "gold": "A",
            "predicted": "A" if i % 2 else "B",
            "p_top1_at_answer": random.random(),
            "p_gold_at_answer": random.random(),
            "correct": (i % 2 == 0),
            "chain_len": random.randint(5, 200),
        }
        for i in range(40)
    ]
    acts = {3: torch.randn(40, 8), 4: torch.randn(40, 8)}
    out = rank_miscalibration_neurons(rows, acts, top_k=5, fdr_q=0.5, length_binned=True)
    assert isinstance(out, list)


def test_m7_length_binned_falls_back_when_too_few_rows():
    """Too few rows → quietly falls back to non-binned (no crash)."""
    from src.h7_layers import rank_miscalibration_neurons

    rows = [
        {
            "q_id": f"q{i}", "gold": "A", "predicted": "A",
            "p_top1_at_answer": 0.5, "p_gold_at_answer": 0.5,
            "correct": True, "chain_len": 10 + i,
        }
        for i in range(5)  # < n_bins * 3 = 12
    ]
    acts = {3: torch.randn(5, 8)}
    out = rank_miscalibration_neurons(rows, acts, top_k=2, fdr_q=0.5, length_binned=True)
    assert isinstance(out, list)


# ---------------------------------------------------------------------------
# M8 — H1 contrastive set length-matched
# ---------------------------------------------------------------------------


def test_m8_positive_negative_lengths_balanced():
    from src.data import NEGATIVE_VIGNETTES, POSITIVE_VIGNETTES
    pos_w = [len(s.split()) for s in POSITIVE_VIGNETTES]
    neg_w = [len(s.split()) for s in NEGATIVE_VIGNETTES]
    pos_mean = statistics.mean(pos_w)
    neg_mean = statistics.mean(neg_w)
    # Mean word-count difference must be small (< 3 words) so per-position
    # gradient × activation isn't measuring length.
    assert abs(pos_mean - neg_mean) < 3.0, (
        f"POSITIVE mean={pos_mean:.1f} vs NEGATIVE mean={neg_mean:.1f} — "
        "length confound reintroduced. Re-balance NEGATIVE_VIGNETTES."
    )


def test_m8_positive_negative_same_count():
    from src.data import NEGATIVE_VIGNETTES, POSITIVE_VIGNETTES
    assert len(POSITIVE_VIGNETTES) == len(NEGATIVE_VIGNETTES) == 20
