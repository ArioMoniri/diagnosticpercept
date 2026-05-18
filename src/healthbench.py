"""H6 — Benchmark evaluation under interventions.

Runs a large clinical-QA benchmark (default: MedQA-USMLE 4-option, ~1273 test
questions) under multiple intervention conditions:

  * baseline               — no intervention
  * ablate_overconf        — zero out the top-3 H5 overconfidence neurons
  * ablate_halluc          — zero out the top-3 H4 hallucination neurons
  * gate_anchor            — H1 anchor intervention at the best gate

For each (condition, question) we record:
  - the question and gold answer
  - the model's raw greedy completion
  - the parsed letter answer
  - whether it matched gold
  - the max next-token softmax probability at the answer slot (calibration proxy)

Two artifacts per condition:
  - ``results/h6/{condition}.jsonl`` (line-delimited per-question records)
  - ``results/h6/comparison.csv``    (one row per question; columns per condition)

This file is the analytic substrate: same questions, same prompts, model
outputs under different neuron-level interventions, ready for offline diff.

HealthBench (OpenAI 2025) is not currently mirrored on HuggingFace under a
permissive license; MedQA-USMLE is the closest open analog. ``load_dataset``
accepts an override name so this harness can switch to any HF dataset whose
schema matches ``{question, options(dict[A..D]), answer_idx or answer}``.
"""
from __future__ import annotations

import csv
import json
import re
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .hooks import anchor_intervention, constant_intervention, zero_mlp_intervention
from .model import LoadedModel, clear_h


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


@dataclass
class MCQItem:
    """One multiple-choice question normalized across source datasets."""

    q_id: str
    question: str
    options: Dict[str, str]    # {"A": "...", "B": "...", ...}
    gold: str                  # "A" | "B" | "C" | "D" | "E"
    source: str = ""


def _letter_for_idx(i: int) -> str:
    return chr(ord("A") + int(i))


def load_medqa(
    name: str = "GBaker/MedQA-USMLE-4-options-hf",
    split: str = "test",
    n: Optional[int] = None,
    seed: int = 0,
) -> List[MCQItem]:
    """Load a clinical MCQ benchmark from HuggingFace.

    Default: MedQA-USMLE 4-option test split (~1273 questions). Pass ``n`` to
    sub-sample (random, seeded). The schema-agnostic adapter handles a few
    common HF variants:

      * GBaker/MedQA-USMLE-4-options-hf  → question, options, answer_idx
      * bigbio/med_qa                    → question, options(list), answer(letter)
      * openlifescienceai/medmcqa        → question, opa/opb/opc/opd, cop(int)
    """
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    items: List[MCQItem] = []
    for i, row in enumerate(ds):
        opts: Dict[str, str] = {}
        gold = ""
        if "options" in row and isinstance(row["options"], dict):
            opts = {k: str(v) for k, v in row["options"].items() if v}
            if "answer_idx" in row:
                gold = str(row["answer_idx"]).strip()[:1].upper()
            elif "answer" in row:
                # Some variants store the answer text; resolve back to a letter.
                a = str(row["answer"]).strip()
                for k, v in opts.items():
                    if v.strip() == a:
                        gold = k.upper()
                        break
        elif "options" in row and isinstance(row["options"], list):
            opts = {_letter_for_idx(j): str(v) for j, v in enumerate(row["options"])}
            if "answer" in row:
                a = str(row["answer"]).strip()[:1].upper()
                gold = a if a in opts else ""
        elif all(k in row for k in ("opa", "opb", "opc", "opd")):
            opts = {"A": row["opa"], "B": row["opb"], "C": row["opc"], "D": row["opd"]}
            cop = row.get("cop", -1)
            if isinstance(cop, int) and 0 <= cop < 4:
                gold = _letter_for_idx(cop)
        else:
            continue

        if not opts or not gold or gold not in opts:
            continue
        items.append(MCQItem(
            q_id=f"{name}#{split}#{i}",
            question=str(row.get("question", "")),
            options=opts,
            gold=gold,
            source=name,
        ))

    if n is not None and n < len(items):
        import random
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


# ---------------------------------------------------------------------------
# Prompting + parsing
# ---------------------------------------------------------------------------


_PROMPT_TEMPLATE = (
    "You are a medical expert answering a USMLE-style multiple-choice question. "
    "First reason concisely through the key findings (1–3 sentences), then "
    "give your final answer as a single letter.\n\n"
    "Question: {question}\n"
    "{options_block}\n\n"
    "Format your response exactly as:\n"
    "Reasoning: <your reasoning>\n"
    "Answer: <single letter>\n"
)


def render_prompt(item: MCQItem) -> str:
    block = "\n".join(f"{k}. {v}" for k, v in sorted(item.options.items()))
    return _PROMPT_TEMPLATE.format(question=item.question.strip(), options_block=block)


_LETTER_RE = re.compile(r"\b([A-E])\b")
_ANSWER_LINE_RE = re.compile(r"answer\s*[:\-]\s*\(?([A-E])\)?", re.IGNORECASE)
_REASONING_RE = re.compile(
    r"reasoning\s*[:\-]\s*(.+?)(?:\n\s*answer|$)", re.IGNORECASE | re.DOTALL
)


def parse_letter(text: str, valid: Sequence[str]) -> Optional[str]:
    """Extract the parsed answer letter.

    Order of preference:
      1. Match an explicit ``Answer: X`` (or ``Answer - X``) line.
      2. Fall back to the first standalone letter in the text.
    Restricted to ``valid`` (e.g. ``["A","B","C","D"]`` rejects E when only 4
    options exist).
    """
    valid_set = {v.upper() for v in valid}
    m = _ANSWER_LINE_RE.search(text)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()
    for m in _LETTER_RE.finditer(text.upper()):
        if m.group(1) in valid_set:
            return m.group(1)
    return None


def parse_reasoning(text: str) -> str:
    """Extract the reasoning text (before the ``Answer:`` line) if present."""
    m = _REASONING_RE.search(text)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# One question, one condition
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRow:
    q_id: str
    condition: str
    question: str
    options: Dict[str, str]
    gold: str
    predicted: Optional[str]
    correct: bool
    reasoning: str          # parsed reasoning chain (before "Answer:")
    raw_output: str
    p_top1: float           # max softmax probability at the first generated token
    p_gold_letter: float    # probability of the gold-letter token (calibration on the answer)


def _letter_token_ids(tokenizer, letters: Sequence[str]) -> Dict[str, int]:
    """First-token ID for " A", " B", ... (leading space matches assistant prefix)."""
    out: Dict[str, int] = {}
    for L in letters:
        ids = tokenizer(" " + L, add_special_tokens=False).input_ids
        if ids:
            out[L] = ids[0]
    return out


@torch.no_grad()
def run_one(
    lm: LoadedModel,
    item: MCQItem,
    condition: str,
    intervention_ctx=None,
    max_new_tokens: int = 220,
) -> BenchmarkRow:
    """One forward+generate under a single intervention context.

    Generates up to ``max_new_tokens`` so the model can lay out its reasoning
    *before* committing to a letter (the prompt template asks for both).
    """
    tok = lm.tokenizer
    prompt = render_prompt(item)
    enc = tok(prompt, return_tensors="pt").to(lm.device)

    ctx = intervention_ctx if intervention_ctx is not None else nullcontext()
    with ctx:
        gen = lm.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
    clear_h(lm.layers)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    first_probs = F.softmax(gen.scores[0][0].float(), dim=-1)
    p_top1 = float(first_probs.max())

    letter_ids = _letter_token_ids(tok, list(item.options.keys()))
    gold_id = letter_ids.get(item.gold)
    p_gold = float(first_probs[gold_id]) if gold_id is not None else 0.0

    raw = tok.decode(
        gen.sequences[0, enc.input_ids.shape[1]:], skip_special_tokens=True
    )
    predicted = parse_letter(raw, list(item.options.keys()))
    reasoning = parse_reasoning(raw)
    correct = (predicted is not None and predicted == item.gold)

    return BenchmarkRow(
        q_id=item.q_id, condition=condition,
        question=item.question, options=item.options, gold=item.gold,
        predicted=predicted, correct=correct,
        reasoning=reasoning, raw_output=raw,
        p_top1=p_top1, p_gold_letter=p_gold,
    )


# ---------------------------------------------------------------------------
# Intervention composer
# ---------------------------------------------------------------------------


@contextmanager
def compose(*ctx_factories: Callable[[], Any]):
    """Enter several context managers; each factory returns a fresh CM.

    Each call returns a *new* CM so we don't risk re-entering a closed
    contextmanager — important because ``constant_intervention`` registers a
    forward pre-hook and unhooks on exit.
    """
    with ExitStack() as stack:
        for factory in ctx_factories:
            stack.enter_context(factory())
        yield


def ablate_neurons_factory(layers, neuron_specs: Sequence[Dict[str, int]]):
    """Returns a *factory* that builds a composed ``constant_intervention(0.0)``
    over a list of ``{layer, neuron}`` records.
    """
    factories = [
        (lambda L=spec["layer"], N=spec["neuron"]:
            constant_intervention(layers, N, 0.0, L))
        for spec in neuron_specs
    ]
    def _build():
        return compose(*factories)
    return _build


def anchor_factory(layers, layer: int, neuron: int, m_star: float, d: float, k: float = 1.0):
    def _build():
        return anchor_intervention(layers, neuron, m_star, d, layer, k=k)
    return _build


def zero_mlp_factory(layers, layer_indices: Sequence[int]):
    """Factory: zero the MLP output at the given layer(s) — H3 critical-layer ablation."""
    def _build():
        return zero_mlp_intervention(layers, list(layer_indices))
    return _build


# ---------------------------------------------------------------------------
# Run a full benchmark over conditions
# ---------------------------------------------------------------------------


def run_conditions(
    lm: LoadedModel,
    items: Sequence[MCQItem],
    conditions: Dict[str, Optional[Callable[[], Any]]],
    out_dir: Path,
    save_every: int = 25,
) -> Dict[str, List[BenchmarkRow]]:
    """Run ``items`` under each named condition; persist after every ``save_every``.

    ``conditions[name]`` is either ``None`` (baseline) or a *factory* that
    returns a fresh context manager when called. The factory pattern lets us
    re-enter the same intervention spec for every question without recycling
    a closed CM.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, List[BenchmarkRow]] = {}

    for cond_name, factory in conditions.items():
        results: List[BenchmarkRow] = []
        jsonl_path = out_dir / f"{cond_name}.jsonl"
        # Resume: if jsonl exists with N records, skip those.
        already = 0
        if jsonl_path.exists():
            with jsonl_path.open() as f:
                for line in f:
                    if line.strip():
                        results.append(BenchmarkRow(**json.loads(line)))
            already = len(results)
            print(f"[{cond_name}] resuming, {already} rows on disk.")
        else:
            jsonl_path.write_text("")  # touch

        for i, item in enumerate(tqdm(items[already:], desc=cond_name, leave=False)):
            ctx = factory() if factory is not None else None
            row = run_one(lm, item, cond_name, intervention_ctx=ctx)
            results.append(row)
            with jsonl_path.open("a") as f:
                f.write(json.dumps(asdict(row)) + "\n")
            if (i + 1) % save_every == 0:
                pass  # already streaming; placeholder for future checkpointing

        all_results[cond_name] = results

    # Write side-by-side CSV across conditions.
    _write_comparison_csv(items, all_results, out_dir / "comparison.csv")
    _write_summary(all_results, out_dir / "summary.json")
    return all_results


def _write_comparison_csv(items, all_results, path: Path) -> None:
    by_id: Dict[str, Dict[str, BenchmarkRow]] = {}
    for cond, rows in all_results.items():
        for r in rows:
            by_id.setdefault(r.q_id, {})[cond] = r
    conds = list(all_results.keys())

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["q_id", "question", "gold"]
        for c in conds:
            header += [f"{c}_pred", f"{c}_correct", f"{c}_p_top1",
                       f"{c}_p_gold", f"{c}_reasoning"]
        w.writerow(header)
        for item in items:
            row_records = by_id.get(item.q_id, {})
            row = [item.q_id, item.question, item.gold]
            for c in conds:
                r = row_records.get(c)
                if r:
                    row += [
                        r.predicted or "", int(r.correct),
                        f"{r.p_top1:.4f}", f"{r.p_gold_letter:.4f}",
                        r.reasoning,
                    ]
                else:
                    row += ["", "", "", "", ""]
            w.writerow(row)


def _write_summary(all_results, path: Path) -> None:
    summary = {}
    for cond, rows in all_results.items():
        n = len(rows)
        acc = sum(int(r.correct) for r in rows) / max(1, n)
        mean_p_top1 = sum(r.p_top1 for r in rows) / max(1, n)
        mean_p_gold = sum(r.p_gold_letter for r in rows) / max(1, n)
        # Calibration brier-style: distance between p_top1 and correctness.
        brier = sum((r.p_top1 - int(r.correct)) ** 2 for r in rows) / max(1, n)
        summary[cond] = {
            "n": n,
            "accuracy": acc,
            "mean_p_top1": mean_p_top1,
            "mean_p_gold_letter": mean_p_gold,
            "brier_score": brier,
        }
    path.write_text(json.dumps(summary, indent=2))
