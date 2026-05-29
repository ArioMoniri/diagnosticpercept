"""Regression test: every H3 pair must have token-matched clean/corrupted
prompts. The patching layer raises if they don't match (since left-padding
zeros silently corrupts the routing measurement), so a CI guard here keeps
the data set honest as we add pairs.
"""
from __future__ import annotations

import pytest

from src.data import H3_PAIRS


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_every_pair_has_equal_token_count(tok):
    """Each (clean, corrupted) pair must tokenize to the same length."""
    mismatches = []
    for p in H3_PAIRS:
        a = len(tok(p.clean_prompt).input_ids)
        b = len(tok(p.corrupted_prompt).input_ids)
        if a != b:
            mismatches.append(f"{p.pair_id}: clean={a} corrupt={b}")
    assert not mismatches, (
        "Token-mismatched H3 pairs (patching would raise at runtime):\n  "
        + "\n  ".join(mismatches)
    )


def test_clean_and_corrupted_dx_tokens_distinct(tok):
    """Sanity: the diagnosis labels themselves must produce distinct first
    tokens, otherwise the logit-diff metric is degenerate."""
    for p in H3_PAIRS:
        c = tok(" " + p.clean_dx, add_special_tokens=False).input_ids
        x = tok(" " + p.corrupted_dx, add_special_tokens=False).input_ids
        assert c and x, f"{p.pair_id}: empty tokenization for dx label"
        assert c[0] != x[0], (
            f"{p.pair_id}: clean_dx {p.clean_dx!r} and corrupted_dx "
            f"{p.corrupted_dx!r} share the same first token id"
        )
