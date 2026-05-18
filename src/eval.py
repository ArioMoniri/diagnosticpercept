"""Evaluation utilities: hedge detector, disease-injection judge, logit-diff helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hedge detection (H1)
# ---------------------------------------------------------------------------

COMMITMENT_PATTERNS = [
    r"\bthe diagnosis is\b",
    r"\bdiagnosis:\b",
    r"\bmost likely\b",
    r"\bconsistent with\b",
    r"\bthis is\b",
    r"\b(I21|E11|J18|J45|A41|F32)\b",
]

HEDGE_PATTERNS = [
    r"\bcould be\b",
    r"\bdifferential\b",
    r"\bpossibilities\b",
    r"\bconsider\b",
    r"\bmight be\b",
    r"\bpossibly\b",
    r"\bdepends on\b",
    r"\bmay (?:represent|indicate|suggest)\b",
    r"\bdifferential diagnosis\b",
]


@dataclass
class HedgeScore:
    """Counts of commitment-style vs hedge-style markers in a generation."""
    commitment_hits: int
    hedge_hits: int
    is_hedging: bool


def score_hedging(text: str) -> HedgeScore:
    """Heuristic hedge detector — H1 success means flipping commitment → hedge."""
    tl = text.lower()
    c = sum(1 for p in COMMITMENT_PATTERNS if re.search(p, tl))
    h = sum(1 for p in HEDGE_PATTERNS if re.search(p, tl))
    return HedgeScore(commitment_hits=c, hedge_hits=h, is_hedging=h > c)


# ---------------------------------------------------------------------------
# Disease-injection judge (H2)
# ---------------------------------------------------------------------------


@dataclass
class InjectionScore:
    """H2 judge: does an amplified-on-benign-prompt generation inject the disease?"""
    mentions: bool
    coherent: bool          # rough proxy: generation length, no degenerate repetition
    relevant: bool          # mentions but still references the benign prompt subject


def score_injection(generation: str, disease_keywords: Sequence[str], prompt_keywords: Sequence[str]) -> InjectionScore:
    """Lightweight version of the paper's M / M+C / M+C+P judge.

    Coherence ≈ generation has at least 8 words and isn't dominated by a single
    repeated 4-gram. Relevance ≈ at least one prompt keyword still appears.
    """
    text = generation.lower()
    mentions = any(k.lower() in text for k in disease_keywords)
    words = text.split()
    coherent = len(words) >= 8
    if coherent:
        ngrams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
        if ngrams:
            top = max(ngrams.count(g) for g in set(ngrams))
            if top > 3:
                coherent = False
    relevant = any(k.lower() in text for k in prompt_keywords) if prompt_keywords else True
    return InjectionScore(mentions=mentions, coherent=coherent, relevant=relevant and mentions)


# ---------------------------------------------------------------------------
# Logit diff (H3 helper, also useful for ad-hoc inspection)
# ---------------------------------------------------------------------------


@torch.no_grad()
def logit_diff_last(logits: torch.Tensor, pos_tid: int, neg_tid: int) -> float:
    """``logits[..., last, pos_tid] − logits[..., last, neg_tid]`` as a float."""
    last = logits[:, -1, :].float()
    return float(last[..., pos_tid].mean() - last[..., neg_tid].mean())


@torch.no_grad()
def target_logprob_last(logits: torch.Tensor, target_ids: Sequence[int]) -> float:
    """``log Σ_t softmax(logits[-1])[t]`` for ``t ∈ target_ids``."""
    last = logits[:, -1, :].float()
    lp = F.log_softmax(last, dim=-1)
    target = torch.as_tensor(target_ids, device=lp.device)
    return float(torch.logsumexp(lp[..., target], dim=-1).mean())
