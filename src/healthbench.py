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

from .hooks import (
    additive_intervention, anchor_intervention, constant_intervention,
    zero_mlp_intervention,
)
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
    skipped: Dict[str, int] = {}

    def _normalize_options(opts_field) -> Dict[str, str]:
        """Accept dict, list, or HF Features-like dict and return {LETTER: text}."""
        if opts_field is None:
            return {}
        if isinstance(opts_field, dict):
            out = {}
            for k, v in opts_field.items():
                if v is None or str(v).strip() == "":
                    continue
                ks = str(k).strip()
                # Some schemas use "A"/"B"/...; others "0"/"1"/...
                if ks.isdigit():
                    ks = _letter_for_idx(int(ks))
                out[ks.upper()] = str(v).strip()
            return out
        if isinstance(opts_field, list):
            return {_letter_for_idx(j): str(v).strip() for j, v in enumerate(opts_field) if v}
        return {}

    for i, row in enumerate(ds):
        opts: Dict[str, str] = _normalize_options(row.get("options"))
        gold = ""
        question = str(row.get("question", "")).strip()

        # Schema C: MedMCQA-style flat opa/opb/opc/opd if no `options` block.
        if not opts and all(k in row for k in ("opa", "opb", "opc", "opd")):
            opts = {"A": str(row["opa"]), "B": str(row["opb"]),
                    "C": str(row["opc"]), "D": str(row["opd"])}
            cop = row.get("cop", -1)
            if isinstance(cop, int) and 0 <= cop < 4:
                gold = _letter_for_idx(cop)

        # Schema D: SWAG-style (used by GBaker/MedQA-USMLE-4-options-hf).
        # Fields: sent1 / sent2 / ending0..ending3 / label (int 0-3).
        if not opts and "ending0" in row:
            endings = [row.get(f"ending{j}") for j in range(5)]
            endings = [e for e in endings if e is not None and str(e).strip()]
            if endings:
                opts = {_letter_for_idx(j): str(e).strip() for j, e in enumerate(endings)}
                if not question:
                    s1 = str(row.get("sent1", "")).strip()
                    s2 = str(row.get("sent2", "")).strip()
                    question = (s1 + ("\n" + s2 if s2 else "")).strip()
                label = row.get("label")
                if isinstance(label, int) and 0 <= label < len(opts):
                    gold = _letter_for_idx(label)

        # Try several fields for gold.
        if not gold:
            for key in ("answer_idx", "answerKey", "answer_letter", "label"):
                if key in row and row[key] is not None:
                    raw = str(row[key]).strip()
                    if raw and raw[0].upper() in opts:
                        gold = raw[0].upper()
                        break
                    if raw.isdigit() and 0 <= int(raw) < len(opts):
                        gold = _letter_for_idx(int(raw))
                        break

        # Last-ditch: match by answer *text*.
        if not gold and "answer" in row and row["answer"] is not None:
            a = str(row["answer"]).strip()
            for k, v in opts.items():
                if v.strip() == a:
                    gold = k
                    break
            # Maybe the answer field is just a letter.
            if not gold and a[:1].upper() in opts:
                gold = a[:1].upper()

        if not opts:
            skipped["no_options"] = skipped.get("no_options", 0) + 1; continue
        if not gold:
            skipped["no_gold"] = skipped.get("no_gold", 0) + 1; continue
        if gold not in opts:
            skipped["gold_not_in_opts"] = skipped.get("gold_not_in_opts", 0) + 1; continue

        items.append(MCQItem(
            q_id=f"{name}#{split}#{i}",
            question=question or str(row.get("question", "")),
            options=opts,
            gold=gold,
            source=name,
        ))

    if not items:
        # Surface enough info to debug a schema mismatch without a second run.
        sample = dict(ds[0]) if len(ds) else {}
        keys = list(sample.keys())
        preview = {k: (str(sample[k])[:120] if sample[k] is not None else None) for k in keys[:6]}
        raise RuntimeError(
            f"load_medqa: 0 items parsed from {name!r} split={split!r}. "
            f"Skipped reasons: {skipped}. First-row keys: {keys}. "
            f"First-row preview: {preview}"
        )

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
    # Calibration at the FIRST generated token (≈ "Reasoning") — not very
    # informative under the reasoning prompt; kept for backward compat.
    p_top1_first: float
    p_gold_first: float
    # Calibration at the token *immediately after* "Answer:" — i.e. the
    # distribution that actually selected the answer letter. This is the
    # meaningful confidence measurement under chain-of-thought.
    p_top1_at_answer: float
    p_gold_at_answer: float
    answer_pos_found: bool   # whether we located "Answer:" in the chain

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkRow":
        """Schema-tolerant constructor — fills missing fields with defaults.

        Lets `run_conditions` resume from jsonl written by an earlier commit
        that lacked some fields. Caught by ml-developer review (C6).
        """
        defaults = {
            "p_top1_first": 0.0, "p_gold_first": 0.0,
            "p_top1_at_answer": 0.0, "p_gold_at_answer": 0.0,
            "answer_pos_found": False, "reasoning": "",
            "raw_output": "", "predicted": None,
        }
        kept = {k: d.get(k, defaults.get(k)) for k in cls.__dataclass_fields__}
        return cls(**kept)



def _letter_token_ids(tokenizer, letters: Sequence[str]) -> Dict[str, int]:
    """First-token ID for each letter under common contexts (legacy shape).

    Returned as ``{letter: first_id}`` keeping the leading-space form as the
    canonical one. Most callers want :func:`_letter_token_id_sets` instead.
    """
    out: Dict[str, int] = {}
    for L in letters:
        ids = tokenizer(" " + L, add_special_tokens=False).input_ids
        if ids:
            out[L] = ids[0]
        else:
            ids = tokenizer(L, add_special_tokens=False).input_ids
            if ids:
                out[L] = ids[0]
    return out


def _letter_token_id_sets(tokenizer, letters: Sequence[str]) -> Dict[str, set]:
    """All plausible first-token IDs per letter across surrounding contexts.

    For each letter we probe leading-space (`' A'`), bare (`'A'`), and
    newline-prefixed (`'\\nA'`) forms — different tokenizers emit different
    IDs depending on what preceded the letter. Probability lookup of
    ``p_gold_letter`` should sum over the set so a missed variant doesn't
    silently zero out the gold probability. Caught by ml-developer review.
    """
    out: Dict[str, set] = {}
    for L in letters:
        ids: set = set()
        for ctx in (" ", "", "\n", "\t"):
            tok_ids = tokenizer(ctx + L, add_special_tokens=False).input_ids
            if tok_ids:
                ids.add(tok_ids[0])
        if ids:
            out[L] = ids
    return out


_ANSWER_END_RE = re.compile(r"answer\s*[:\-]\s*$", re.IGNORECASE | re.DOTALL)


def _find_answer_token_pos(tokenizer, generated_ids: torch.Tensor, max_search: int = 800) -> Optional[int]:
    """Locate the position (in ``generated_ids``) whose logit distribution
    *chose the answer letter* — i.e. the token immediately after "Answer:".

    Decoding token-by-token and matching the accumulated text against
    ``answer\\s*[:\\-]\\s*$`` handles every tokenizer quirk: whether
    "Answer:" is one token or three, whether there is a leading space, etc.

    Returns the index ``i`` into ``generated_ids`` such that
    ``gen.scores[i]`` is the logit distribution that produced the answer
    letter, and ``generated_ids[i]`` is the letter token itself. Returns
    ``None`` if "Answer:" never appears in the generation.
    """
    # Cumulative decode of generated_ids[:i+1] — robust to BPE byte-fallback
    # tokens (e.g. multi-byte chars in clinical text like —, μ, β) that
    # split across token boundaries. Per-token decode could lose those.
    # Caught by ml-developer review 2026-05-29.
    upper = min(int(generated_ids.shape[0]), max_search)
    for i in range(upper):
        text = tokenizer.decode(generated_ids[:i + 1], skip_special_tokens=True)
        if _ANSWER_END_RE.search(text):
            # The next token (i+1) was emitted *by* the logit at position i+1,
            # which we have in gen.scores[i+1]. But the letter token itself is
            # generated_ids[i+1]; bounds-check first.
            if i + 1 < int(generated_ids.shape[0]):
                return i + 1
            return None
    return None


@torch.no_grad()
def run_one(
    lm: LoadedModel,
    item: MCQItem,
    condition: str,
    intervention_ctx=None,
    max_new_tokens: int = 512,
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

    # ---- Calibration at first token (legacy, ≈ "Reasoning") ----
    first_probs = F.softmax(gen.scores[0][0].float(), dim=-1)
    p_top1_first = float(first_probs.max())

    # Use the SET version of letter ids so we sum probability across leading-
    # space, bare, newline-prefixed token variants. Single-id lookup misses
    # the gold-token when the tokenizer emitted the other form.
    letter_id_sets = _letter_token_id_sets(tok, list(item.options.keys()))
    gold_ids = letter_id_sets.get(item.gold, set())
    p_gold_first = float(sum(first_probs[i].item() for i in gold_ids))

    # ---- Calibration AT the answer letter (the meaningful signal) ----
    generated_ids = gen.sequences[0, enc.input_ids.shape[1]:]
    ans_pos = _find_answer_token_pos(tok, generated_ids)
    if ans_pos is not None and ans_pos < len(gen.scores):
        ans_probs = F.softmax(gen.scores[ans_pos][0].float(), dim=-1)
        p_top1_at_answer = float(ans_probs.max())
        p_gold_at_answer = float(sum(ans_probs[i].item() for i in gold_ids))
        answer_pos_found = True
    else:
        # Fall back to first-token if the model didn't honor the format.
        p_top1_at_answer = p_top1_first
        p_gold_at_answer = p_gold_first
        answer_pos_found = False

    raw = tok.decode(generated_ids, skip_special_tokens=True)
    predicted = parse_letter(raw, list(item.options.keys()))
    reasoning = parse_reasoning(raw)
    correct = (predicted is not None and predicted == item.gold)

    row = BenchmarkRow(
        q_id=item.q_id, condition=condition,
        question=item.question, options=item.options, gold=item.gold,
        predicted=predicted, correct=correct,
        reasoning=reasoning, raw_output=raw,
        p_top1_first=p_top1_first, p_gold_first=p_gold_first,
        p_top1_at_answer=p_top1_at_answer, p_gold_at_answer=p_gold_at_answer,
        answer_pos_found=answer_pos_found,
    )
    # Free the generate-time KV cache + scores AFTER all metric reads. This
    # is critical for parallel workers: without it, reasoning chains spike
    # VRAM ~5-8 GB per call and the next gen can OOM. C7 fix: also explicitly
    # delete the scores list (max_new_tokens × [1, vocab] = ~310 MB / call
    # for Qwen3 at 512 new tokens) BEFORE clearing the cache, otherwise the
    # cuda allocator keeps the reserved pool high across questions.
    if hasattr(gen, "scores") and gen.scores is not None:
        gen.scores = None
    del gen, generated_ids, first_probs
    if torch.cuda.is_available():
        import gc; gc.collect()
        torch.cuda.empty_cache()
    return row


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


def additive_shift_factory(layers, shifts: Sequence[Dict[str, Any]]):
    """Factory that applies an additive offset to each (layer, neuron) pair.

    ``shifts`` is a list of ``{layer, neuron, amount}`` records. Used for the
    H7 "anchor toward calibrated profile" intervention where the offset is
    ``mean_calib − mean_overconf`` — a soft shift that preserves per-token
    context, unlike ``constant_intervention(value=0)`` which fully overwrites.
    """
    factories = [
        (lambda L=s["layer"], N=s["neuron"], A=float(s["amount"]):
            additive_intervention(layers, N, A, L))
        for s in shifts
    ]
    def _build():
        return compose(*factories)
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
                        results.append(BenchmarkRow.from_dict(json.loads(line)))
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
            header += [
                f"{c}_pred", f"{c}_correct",
                f"{c}_p_top1_first",  f"{c}_p_gold_first",
                f"{c}_p_top1_answer", f"{c}_p_gold_answer",
                f"{c}_answer_found",  f"{c}_reasoning",
            ]
        w.writerow(header)
        for item in items:
            row_records = by_id.get(item.q_id, {})
            row = [item.q_id, item.question, item.gold]
            for c in conds:
                r = row_records.get(c)
                if r:
                    row += [
                        r.predicted or "", int(r.correct),
                        f"{r.p_top1_first:.4f}", f"{r.p_gold_first:.4f}",
                        f"{r.p_top1_at_answer:.4f}", f"{r.p_gold_at_answer:.4f}",
                        int(r.answer_pos_found), r.reasoning,
                    ]
                else:
                    row += ["", "", "", "", "", "", "", ""]
            w.writerow(row)


def _write_summary(all_results, path: Path) -> None:
    summary = {}
    for cond, rows in all_results.items():
        n = len(rows)
        acc = sum(int(r.correct) for r in rows) / max(1, n)
        # Legacy first-token metrics (kept for backward compat).
        mean_p_first = sum(r.p_top1_first for r in rows) / max(1, n)
        # Answer-position metrics — the actual confidence on the answer letter.
        mean_p_top1_ans = sum(r.p_top1_at_answer for r in rows) / max(1, n)
        mean_p_gold_ans = sum(r.p_gold_at_answer for r in rows) / max(1, n)
        # Brier at the answer position (lower = better-calibrated).
        brier_at_answer = sum(
            (r.p_top1_at_answer - int(r.correct)) ** 2 for r in rows
        ) / max(1, n)
        # Fraction of rows where we found "Answer:" — a sanity check on format adherence.
        answer_found_rate = sum(int(r.answer_pos_found) for r in rows) / max(1, n)
        summary[cond] = {
            "n": n,
            "accuracy": acc,
            "mean_p_top1_first": mean_p_first,
            "mean_p_top1_at_answer": mean_p_top1_ans,
            "mean_p_gold_at_answer": mean_p_gold_ans,
            "brier_at_answer": brier_at_answer,
            "answer_position_found_rate": answer_found_rate,
        }
    path.write_text(json.dumps(summary, indent=2))
