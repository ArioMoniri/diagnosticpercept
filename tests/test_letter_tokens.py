"""Regression tests for letter-token id resolution + answer-position decode.

These bake in the reviewer's findings so a future refactor doesn't quietly
revert them:

- `_letter_token_id_sets` covers leading-space, bare, newline, tab forms.
- `_find_answer_token_pos` uses cumulative decode (robust to BPE byte
  fallback in clinical multi-byte chars).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch

from src.healthbench import _find_answer_token_pos, _letter_token_id_sets


class _FakeTokenizer:
    """Minimal tokenizer that maps strings to deterministic id sequences."""

    def __init__(self, vocab):
        # vocab: list of (text_fragment, token_id) tuples for decode/encode
        self.vocab = list(vocab)

    def __call__(self, text, add_special_tokens=False):
        # Greedy left-to-right longest-match encode.
        ids = []
        rest = text
        while rest:
            best = None
            for frag, tid in self.vocab:
                if rest.startswith(frag) and (best is None or len(frag) > len(best[0])):
                    best = (frag, tid)
            if best is None:
                raise ValueError(f"unable to encode at {rest!r}")
            ids.append(best[1])
            rest = rest[len(best[0]):]
        return MagicMock(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        out = []
        for tid in ids:
            for frag, t in self.vocab:
                if t == tid:
                    out.append(frag); break
        return "".join(out)


def _vocab(*pairs):
    return list(pairs)


def test_letter_token_id_sets_covers_space_bare_newline():
    """Different contexts produce different ids; sets should include them all."""
    tok = _FakeTokenizer(_vocab(
        (" A", 10), ("A", 11), ("\nA", 12), ("\tA", 13),
        (" B", 20), ("B", 21), ("\nB", 22), ("\tB", 23),
    ))
    sets = _letter_token_id_sets(tok, ["A", "B"])
    assert sets["A"] == {10, 11, 12, 13}
    assert sets["B"] == {20, 21, 22, 23}


def test_letter_token_id_sets_handles_collisions():
    """If two contexts map to the same id, the set just contains it once."""
    tok = _FakeTokenizer(_vocab((" A", 10), ("A", 10), ("\nA", 10), ("\tA", 10)))
    sets = _letter_token_id_sets(tok, ["A"])
    assert sets["A"] == {10}


def test_find_answer_token_pos_uses_cumulative_decode():
    """`Answer:` split across tokens should be detected by cumulative decode.

    Per-token decode would have produced an empty fragment for token 1
    (an isolated byte). Cumulative decode reassembles the full text.
    """
    tok = _FakeTokenizer(_vocab(
        ("Reasoning:", 1), (" the", 2), (" diagnosis", 3),
        ("\nAnswer", 4), (":", 5), (" B", 6),
        (" because", 7), (" of", 8),
    ))
    gen_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    pos = _find_answer_token_pos(tok, gen_ids)
    # "Answer:" ends at index 4 (the colon), so next token (5) is " B" position.
    assert pos == 5


def test_find_answer_token_pos_returns_none_when_absent():
    """No `Answer:` in generation → None (no false positive)."""
    tok = _FakeTokenizer(_vocab(
        ("Some", 1), (" text", 2), (" with", 3), (" no", 4),
        (" answer", 5), (" anywhere", 6),
    ))
    gen_ids = torch.tensor([1, 2, 3, 4, 5, 6])
    assert _find_answer_token_pos(tok, gen_ids) is None
