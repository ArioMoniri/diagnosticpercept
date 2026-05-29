"""Consensus-flip + reasoning-letter analyzers over a H6 comparison CSV.

A *consensus-flip case* is a question where:
  - the model's baseline reasoning text mentions the *gold* letter
    (i.e. the model has the right answer in its chain of thought),
  - but the model's *committed* baseline answer is a different letter.

These cases are interesting because the model's "knowledge" is intact, but
something at the *commitment* step (the H1 gate) sends it to the wrong
letter. Per the H1 thesis, ablating / anchoring the gate should fix these
cases disproportionately.

The analyzer here works entirely from the comparison.csv emitted by
``src.healthbench`` — no model needed — so it can be re-run any time.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


_LETTER_MENTION_RE = re.compile(r"\b\(?([A-E])\)?\b")

# M6 (ml-developer review 2026-05-29): a most-frequent-letter heuristic over
# the reasoning text over-calls "consensus flips" — incidental mentions like
# "option B can be ruled out" can outvote a single decisive "the answer is D".
# The stricter regex below matches *commitment* phrases ("the answer is X",
# "best choice is X", "therefore X", "I choose X", "diagnosis is X") and only
# falls back to the frequency heuristic when no commitment phrase fires.
_COMMITMENT_LETTER_RE = re.compile(
    r"(?:the\s+answer\s+is|therefore[, ]+|so\s+i\s+(?:would\s+)?(?:pick|choose|go\s+with)|"
    r"best\s+(?:answer|choice|option)\s+is|i\s+(?:would\s+)?(?:pick|choose|select|go\s+with)|"
    r"(?:my\s+)?diagnosis\s+is|the\s+correct\s+answer\s+is|"
    r"most\s+likely(?:\s+answer)?\s+is|my\s+pick\s+is|answer\s*=\s*)"
    r"\s*\(?([A-E])\)?\b",
    re.IGNORECASE,
)

# iter-6: post-letter commitment phrasings — "D is the correct answer",
# "D is the best choice", "D is my diagnosis", etc. Matches a letter
# followed by the predicate, with the *first* alternation group capturing
# the letter (post-form). Used as a secondary scan when prefix-form misses.
_COMMITMENT_LETTER_POST_RE = re.compile(
    r"\b\(?([A-E])\)?\s+(?:is\s+(?:the\s+)?"
    r"(?:correct|best|right|final|most\s+likely)\s*"
    r"(?:answer|choice|option|diagnosis)|"
    r"is\s+my\s+(?:answer|choice|pick|diagnosis))\b",
    re.IGNORECASE,
)


def implied_letter(reasoning: str, valid_letters: Sequence[str]) -> Optional[str]:
    """Infer the letter the reasoning text *votes for*.

    Order of preference (M6, stricter):
      1. **Commitment-phrase match.** Look for "the answer is X", "I choose X",
         "best choice is X", etc. If any match is in ``valid_letters``, the
         *last* such phrase wins (the model's final commitment within the
         reasoning chain).
      2. **Frequency fallback.** Count all standalone-letter mentions; return
         the most-frequent letter. Ties → ``None``.
    """
    valid = {v.upper() for v in valid_letters}

    # 1. Commitment-phrase match (preferred). Combine prefix-form and
    # post-form regexes and take the *last* hit by string position so
    # late-chain commitments override earlier hedging.
    hits: List = []  # list of (start_pos, letter)
    for m in _COMMITMENT_LETTER_RE.finditer(reasoning):
        L = m.group(1).upper()
        if L in valid:
            hits.append((m.start(), L))
    for m in _COMMITMENT_LETTER_POST_RE.finditer(reasoning):
        L = m.group(1).upper()
        if L in valid:
            hits.append((m.start(), L))
    if hits:
        hits.sort(key=lambda x: x[0])
        return hits[-1][1]

    # 2. Frequency fallback.
    counts: Counter = Counter()
    for m in _LETTER_MENTION_RE.finditer(reasoning):
        L = m.group(1).upper()
        if L in valid:
            counts[L] += 1
    if not counts:
        return None
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


@dataclass
class ConsensusFlipRow:
    """One row of the analyzer output."""
    q_id: str
    gold: str
    baseline_pred: str
    baseline_implied: Optional[str]
    is_consensus_flip: bool
    fixed_by: List[str]                  # list of conditions that fixed it


def analyze(csv_path: Path, conditions: Sequence[str]) -> List[ConsensusFlipRow]:
    """Walk a comparison.csv and tag every row with consensus-flip status +
    which intervention conditions corrected it."""
    out: List[ConsensusFlipRow] = []
    with Path(csv_path).open() as f:
        for r in csv.DictReader(f):
            gold = r.get("gold", "")
            base_pred = r.get("baseline_pred", "")
            base_reasoning = r.get("baseline_reasoning", "")
            # The set of letter labels in this row (typically A,B,C,D).
            valid_letters = ["A", "B", "C", "D", "E"]
            implied = implied_letter(base_reasoning, valid_letters)

            is_flip = (
                implied is not None
                and implied == gold
                and base_pred != gold
            )
            fixed_by: List[str] = []
            if is_flip:
                for c in conditions:
                    if c == "baseline":
                        continue
                    if r.get(f"{c}_pred", "") == gold:
                        fixed_by.append(c)
            out.append(ConsensusFlipRow(
                q_id=r.get("q_id", ""), gold=gold,
                baseline_pred=base_pred, baseline_implied=implied,
                is_consensus_flip=is_flip, fixed_by=fixed_by,
            ))
    return out


def summarize(rows: Sequence[ConsensusFlipRow], conditions: Sequence[str]) -> Dict:
    """Aggregate counts: total flips, fix rate per condition (out of flips),
    and the random-rate baseline (out of all baseline-wrong)."""
    flips = [r for r in rows if r.is_consensus_flip]
    base_wrong = [r for r in rows if r.baseline_pred != r.gold]
    out = {
        "n_total": len(rows),
        "n_baseline_wrong": len(base_wrong),
        "n_consensus_flips": len(flips),
        "consensus_flip_rate": len(flips) / max(1, len(rows)),
        "fix_rates": {},
    }
    for c in conditions:
        if c == "baseline":
            continue
        fixed_on_flips = sum(1 for r in flips if c in r.fixed_by)
        # Random baseline: how often this condition fixes ANY baseline-wrong case.
        base_wrong_qids = {r.q_id for r in base_wrong}
        fixed_on_any = 0
        for r in rows:
            if r.baseline_pred != r.gold and c in r.fixed_by:
                fixed_on_any += 1
        out["fix_rates"][c] = {
            "on_flips": fixed_on_flips,
            "on_flips_rate": fixed_on_flips / max(1, len(flips)),
            "on_any_baseline_wrong": fixed_on_any,
            "on_any_rate": fixed_on_any / max(1, len(base_wrong)),
        }
    return out
