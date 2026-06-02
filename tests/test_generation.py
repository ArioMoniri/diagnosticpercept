"""Tests for the H6 generation-speed fixes: early-stop criterion + fast-path.

The benchmark was hanging ~24 h because run_one generated to max_new_tokens
(512) on every question. These cover the early-stop logic that halts shortly
after "Answer: X" and the fast-path that skips the O(N^2) answer-position
decode when the model never emitted "answer". All CPU-only (no model load).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _CharTok:
    """Decode token-ids-as-codepoints back to text (for the criterion test)."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if int(i) >= 9)


def _feed(crit, prompt_len, text):
    """Append one codepoint per step; return the list of stop decisions."""
    ids = list(range(65, 65 + prompt_len))  # arbitrary prompt tokens
    out = []
    for ch in text:
        ids.append(ord(ch))
        out.append(bool(crit(torch.tensor([ids]))))
    return out


def test_answer_stopping_fires_after_answer_letter():
    from src.healthbench import _AnswerStopping
    crit = _AnswerStopping(_CharTok(), prompt_len=3, tail=64, extra=2)
    text = "Reasoning: chest pain and ST elevation.\nAnswer: B.."
    decisions = _feed(crit, 3, text)
    # Index where the answer letter 'B' is emitted.
    b_idx = text.index("Answer: B") + len("Answer: B") - 1
    # Must NOT stop before the letter is present.
    assert not any(decisions[: b_idx + 1])
    # Must stop within `extra` (=2) tokens after the letter.
    assert any(decisions[b_idx + 1: b_idx + 4])


def test_answer_stopping_never_fires_without_answer():
    from src.healthbench import _AnswerStopping
    crit = _AnswerStopping(_CharTok(), prompt_len=3, tail=64, extra=2)
    decisions = _feed(crit, 3, "Reasoning: the patient likely has pneumonia here.")
    assert not any(decisions)


def test_answer_stopping_ignores_short_generations():
    from src.healthbench import _AnswerStopping
    crit = _AnswerStopping(_CharTok(), prompt_len=3, tail=64, extra=2)
    # gen_len < 2 → always False (avoids decoding a 1-token tail).
    ids = torch.tensor([[65, 66, 67, ord("A")]])  # prompt_len 3 + 1 gen token
    assert crit(ids) is False


def test_build_stopping_returns_list_or_none():
    from src.healthbench import _build_stopping

    class _T:
        def decode(self, ids, skip_special_tokens=True):
            return ""

    s = _build_stopping(_T(), 5)
    # transformers is installed in CI → a StoppingCriteriaList; tolerate None.
    assert s is None or len(s) == 1


def test_default_max_new_tokens_is_modest():
    """The default cap must be well below the old 512 so questions are short."""
    from src.healthbench import _DEFAULT_MAX_NEW_TOKENS
    assert 64 <= _DEFAULT_MAX_NEW_TOKENS <= 320


def test_answer_stop_regex_matches_common_forms():
    from src.healthbench import _ANSWER_STOP_RE
    for s in ("Answer: B", "answer:C", "Answer - D", "Answer:  (A"):
        assert _ANSWER_STOP_RE.search(s), s
    assert not _ANSWER_STOP_RE.search("Reasoning: the answer depends")
