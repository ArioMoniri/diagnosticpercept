"""H8 — Cross-task confidence circuitry.

H5 found "I'm sure" neurons on conversational *prose* (Are you confident?
yes/no). H7 found miscalibration neurons on *MCQ* (answer-letter slot).
The H5 ∩ H7 intersection was empty on the first 1273-question run.

H8 measures both signals on the **same questions** so we can attribute the
difference to task format, not to data drift:

  For each MedQA item:
    A) Run the standard MCQ prompt → record `p_top1_at_answer_mcq`,
       capture per-layer signed-max activations at the answer-letter
       forward (`act_mcq`).
    B) Take the model's MCQ answer. Build a prose attestation prompt:
         "<question>\nThe answer is <letter>: <option_text>.\n
          Are you confident in this answer? Answer yes or no.\nAnswer:"
       Run it → record `p_yes`, capture per-layer signed-max activations
       at the post-"Answer:" forward (`act_prose`).
    C) miscal_mcq   = p_top1_at_answer_mcq − int(correct)
       miscal_prose = p_yes − p_top1_at_answer_mcq   (stated > actual)

  For each neuron:
    r_mcq   = corr(act_mcq[:,n], miscal_mcq)
    r_prose = corr(act_prose[:,n], miscal_prose)

A scatter of (r_mcq, r_prose) across all neurons in a layer band lets us
classify:
  - TASK-GENERAL  : both r large positive — fires on overconfidence in any format
  - MCQ-SPECIFIC  : r_mcq large, r_prose ~ 0 — answer-position confidence
  - PROSE-SPECIFIC: r_prose large, r_mcq ~ 0 — verbal attestation confidence
  - NEITHER       : both ~ 0

Per-layer counts of each category map *where* "I'm sure" rises and how the
two task-circuits relate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .calibration import CONFIDENT_TOKENS, UNSURE_TOKENS, _gather_token_ids
from .healthbench import (
    MCQItem, _find_answer_token_pos, _letter_token_ids, render_prompt,
)
from .model import LoadedModel, clear_h


_PROSE_ATTEST_TEMPLATE = (
    "Question: {question}\n"
    "The answer is {letter}: {option}.\n"
    "Are you confident in this answer? Answer yes or no.\n"
    "Answer:"
)


@dataclass
class XTaskRow:
    q_id: str
    gold: str
    predicted: str
    correct: bool
    p_top1_mcq: float
    p_yes_prose: float
    p_no_prose: float
    miscal_mcq: float
    miscal_prose: float


@dataclass
class XTaskNeuron:
    layer: int
    neuron: int
    r_mcq: float
    r_prose: float
    category: str    # 'general' | 'mcq_only' | 'prose_only' | 'neither'


def _signed_max_at_position(h: torch.Tensor) -> torch.Tensor:
    """``h`` is [1, T, d_ff]. Return the signed-max-abs activation per neuron
    over the sequence length, on CPU, [d_ff]."""
    argmax = h.abs().argmax(dim=1, keepdim=True)
    return h.gather(1, argmax).squeeze(1).squeeze(0).cpu()


@torch.no_grad()
def collect_xtask(
    lm: LoadedModel,
    items: Sequence[MCQItem],
    layer_indices: Optional[Sequence[int]] = None,
    max_new_tokens_mcq: int = 220,
) -> Tuple[List[XTaskRow], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """For each item, run the MCQ + prose attestation forwards and capture
    per-layer signed-max activations from each.

    Returns
    -------
    rows : per-item XTaskRow records (one per item where we found an MCQ
        Answer-position; items where the format wasn't honored are skipped).
    acts_mcq : ``{layer: Tensor[N, d_ff]}`` — activations at the MCQ answer pos.
    acts_prose : ``{layer: Tensor[N, d_ff]}`` — activations at the prose
        attestation answer pos (one token after "Answer:" in the prose run).
    """
    tok = lm.tokenizer
    if layer_indices is None:
        # Wide band: later half of the model.
        layer_indices = list(range(lm.n_layers // 2, lm.n_layers))

    yes_ids = _gather_token_ids(tok, CONFIDENT_TOKENS)
    no_ids = _gather_token_ids(tok, UNSURE_TOKENS)

    rows: List[XTaskRow] = []
    mcq_buf: Dict[int, List[torch.Tensor]] = {L: [] for L in layer_indices}
    prose_buf: Dict[int, List[torch.Tensor]] = {L: [] for L in layer_indices}

    for item in tqdm(items, desc="H8 collect", leave=False):
        # ---------- MCQ forward ----------
        prompt = render_prompt(item)
        enc = tok(prompt, return_tensors="pt").to(lm.device)
        gen = lm.model.generate(
            **enc, max_new_tokens=max_new_tokens_mcq, do_sample=False,
            pad_token_id=tok.pad_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
        clear_h(lm.layers)

        gen_ids = gen.sequences[0, enc.input_ids.shape[1]:]
        ans_pos = _find_answer_token_pos(tok, gen_ids)
        if ans_pos is None:
            continue
        ans_probs = F.softmax(gen.scores[ans_pos][0].float(), dim=-1)
        p_top1_mcq = float(ans_probs.max())

        letter_ids = _letter_token_ids(tok, list(item.options.keys()))
        predicted_letter = ""
        pred_id = int(gen_ids[ans_pos])
        for L, tid in letter_ids.items():
            if tid == pred_id:
                predicted_letter = L
                break
        if not predicted_letter:
            continue
        correct = (predicted_letter == item.gold)

        # Capture MCQ acts at the answer-letter position by re-running the
        # prompt + tokens up to and including the answer letter.
        full_ids = torch.cat([enc.input_ids[0], gen_ids[:ans_pos + 1]]).unsqueeze(0)
        lm.model(input_ids=full_ids, use_cache=False)
        for L in layer_indices:
            mcq_buf[L].append(_signed_max_at_position(lm.layers[L].mlp._h.detach().float()))
        clear_h(lm.layers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------- Prose attestation forward ----------
        prose_prompt = _PROSE_ATTEST_TEMPLATE.format(
            question=item.question.strip(),
            letter=predicted_letter,
            option=item.options.get(predicted_letter, "").strip(),
        )
        enc2 = tok(prose_prompt, return_tensors="pt").to(lm.device)
        out2 = lm.model(input_ids=enc2.input_ids, use_cache=False)
        last_logits = out2.logits[0, -1].float()
        probs = F.softmax(last_logits, dim=-1)
        p_yes = float(probs[torch.as_tensor(yes_ids, device=probs.device)].sum())
        p_no = float(probs[torch.as_tensor(no_ids, device=probs.device)].sum())

        for L in layer_indices:
            prose_buf[L].append(_signed_max_at_position(lm.layers[L].mlp._h.detach().float()))
        clear_h(lm.layers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows.append(XTaskRow(
            q_id=item.q_id, gold=item.gold,
            predicted=predicted_letter, correct=correct,
            p_top1_mcq=p_top1_mcq,
            p_yes_prose=p_yes, p_no_prose=p_no,
            miscal_mcq=p_top1_mcq - int(correct),
            miscal_prose=p_yes - p_top1_mcq,
        ))

    acts_mcq = {L: torch.stack(mcq_buf[L], dim=0) for L in layer_indices if mcq_buf[L]}
    acts_prose = {L: torch.stack(prose_buf[L], dim=0) for L in layer_indices if prose_buf[L]}
    return rows, acts_mcq, acts_prose


def _pearson(A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Column-wise Pearson r of [N, d] against [N]; returns [d]."""
    a_mean = A.mean(dim=0)
    a_var = ((A - a_mean) ** 2).sum(dim=0).clamp_min(1e-8)
    y_mean = y.mean()
    y_var = ((y - y_mean) ** 2).sum().clamp_min(1e-8)
    cov = ((A - a_mean) * (y - y_mean).unsqueeze(1)).sum(dim=0)
    return cov / torch.sqrt(a_var * y_var)


def classify_neurons(
    rows: Sequence[XTaskRow],
    acts_mcq: Dict[int, torch.Tensor],
    acts_prose: Dict[int, torch.Tensor],
    r_threshold: float = 0.15,
) -> List[XTaskNeuron]:
    """Per-neuron Pearson r in both tasks, plus a category tag.

    ``r_threshold`` defines the "significant" cutoff. At N=200+ a sample
    correlation of ±0.15 is roughly two-sigma (p ~ 0.03).
    """
    miscal_mcq = torch.tensor([r.miscal_mcq for r in rows], dtype=torch.float32)
    miscal_prose = torch.tensor([r.miscal_prose for r in rows], dtype=torch.float32)

    out: List[XTaskNeuron] = []
    for L, A_mcq in acts_mcq.items():
        A_prose = acts_prose.get(L)
        if A_prose is None:
            continue
        r_mcq = _pearson(A_mcq, miscal_mcq)
        r_prose = _pearson(A_prose, miscal_prose)
        for n in range(r_mcq.numel()):
            rm = float(r_mcq[n])
            rp = float(r_prose[n])
            if rm >= r_threshold and rp >= r_threshold:
                cat = "general"
            elif rm >= r_threshold and abs(rp) < r_threshold:
                cat = "mcq_only"
            elif rp >= r_threshold and abs(rm) < r_threshold:
                cat = "prose_only"
            else:
                cat = "neither"
            out.append(XTaskNeuron(layer=int(L), neuron=int(n),
                                    r_mcq=rm, r_prose=rp, category=cat))
    return out


def category_summary(neurons: Sequence[XTaskNeuron]) -> Dict:
    """Per-layer histogram of (general / mcq_only / prose_only / neither)."""
    summary: Dict = {}
    for n in neurons:
        layer = summary.setdefault(n.layer, {"general": 0, "mcq_only": 0,
                                              "prose_only": 0, "neither": 0})
        layer[n.category] += 1
    return summary
