"""H7 — Calibration-failure layers at MedQA scale.

H5 looks for overconfidence neurons over 20 hard cases by Pearson r between
activation and ``p_yes − p_dx``. Small N. H7 generalizes the same idea to
the 1000+ MedQA test set using a calibration signal that is well-defined
on multiple choice:

    miscal_i = p_top1_at_answer_i − int(correct_i)

i.e. how *over-confident* the model is on question i (high p_top1 but wrong
→ large positive; high p_top1 and right → ~0; low p_top1 and right → small
negative). Across N questions, per-layer/per-neuron Pearson correlation
between activation at the answer-position forward and ``miscal`` identifies
the circuits that encode "I'm confident" *regardless* of whether they should.

The MedQA-scale signal is far stronger than 20 hard cases — even small
effect sizes (r ~ 0.1) are detectable at N=500+.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .healthbench import (
    MCQItem, _find_answer_token_pos, _letter_token_ids, render_prompt,
)
from .model import LoadedModel, clear_h


@dataclass
class MiscalNeuron:
    layer: int
    neuron: int
    pearson_r: float
    n: int
    mean_act_overconf: float    # activation on high-|miscal| cases
    mean_act_calib: float       # activation on near-zero-miscal cases


@torch.no_grad()
def collect_answer_position_acts(
    lm: LoadedModel,
    items: Sequence[MCQItem],
    layer_indices: Optional[Sequence[int]] = None,
    max_new_tokens: int = 220,
) -> Tuple[List[Dict[str, float]], Dict[int, torch.Tensor]]:
    """Run each item, find the answer-position, capture per-layer
    signed-max activations *up to and including* the answer-position
    forward, and the calibration measurement.

    Returns
    -------
    rows : list of dicts, one per item, containing q_id, gold, predicted,
        p_top1_at_answer, p_gold_at_answer, correct.
    acts : {layer_idx: Tensor[N, d_ff]} stacked across items.
    """
    tok = lm.tokenizer
    if layer_indices is None:
        # Later half of the model — that's where commitment/confidence live.
        layer_indices = list(range(lm.n_layers // 2, lm.n_layers))

    rows: List[Dict[str, float]] = []
    acts_per_layer: Dict[int, List[torch.Tensor]] = {L: [] for L in layer_indices}

    for item in tqdm(items, desc="H7 collect", leave=False):
        prompt = render_prompt(item)
        enc = tok(prompt, return_tensors="pt").to(lm.device)
        gen = lm.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
        clear_h(lm.layers)

        generated_ids = gen.sequences[0, enc.input_ids.shape[1]:]
        ans_pos = _find_answer_token_pos(tok, generated_ids)
        if ans_pos is None:
            # Skip questions where the model didn't honor the format —
            # we can't measure calibration at a position that doesn't exist.
            continue

        ans_probs = F.softmax(gen.scores[ans_pos][0].float(), dim=-1)
        p_top1 = float(ans_probs.max())
        letter_ids = _letter_token_ids(tok, list(item.options.keys()))
        gold_id = letter_ids.get(item.gold)
        p_gold = float(ans_probs[gold_id]) if gold_id is not None else 0.0

        predicted_id = int(generated_ids[ans_pos])
        # Map predicted token id back to a letter (or "" if it isn't one).
        predicted_letter = ""
        for L, tid in letter_ids.items():
            if tid == predicted_id:
                predicted_letter = L; break
        correct = (predicted_letter == item.gold)

        # Re-run a *single* forward over the prompt + everything up to and
        # including the answer-position token to capture h on each layer at
        # the moment the answer letter was emitted. This is the activation
        # snapshot we correlate with miscalibration.
        full_ids = torch.cat([enc.input_ids[0], generated_ids[:ans_pos + 1]]).unsqueeze(0)
        lm.model(input_ids=full_ids, use_cache=False)
        for L in layer_indices:
            h = lm.layers[L].mlp._h.detach().float()  # [1, T, d_ff]
            argmax = h.abs().argmax(dim=1, keepdim=True)
            signed = h.gather(1, argmax).squeeze(1).squeeze(0).cpu()  # [d_ff]
            acts_per_layer[L].append(signed)
        clear_h(lm.layers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows.append({
            "q_id": item.q_id, "gold": item.gold,
            "predicted": predicted_letter,
            "p_top1_at_answer": p_top1,
            "p_gold_at_answer": p_gold,
            "correct": correct,
        })

    stacked = {L: torch.stack(acts_per_layer[L], dim=0) for L in layer_indices if acts_per_layer[L]}
    return rows, stacked


def rank_miscalibration_neurons(
    rows: Sequence[Dict[str, float]],
    acts: Dict[int, torch.Tensor],
    top_k: int = 20,
) -> List[MiscalNeuron]:
    """Per-neuron Pearson r between activation and ``miscal = p_top1 − correct``."""
    miscal = torch.tensor(
        [r["p_top1_at_answer"] - float(r["correct"]) for r in rows], dtype=torch.float32
    )  # [N]
    m_mean = miscal.mean()
    m_var = ((miscal - m_mean) ** 2).sum().clamp_min(1e-8)

    # Define "overconfident" subset = top quartile by miscal.
    n = len(miscal)
    q = max(1, n // 4)
    sorted_idx = torch.argsort(miscal, descending=True)
    high_idx = sorted_idx[:q].tolist()
    low_idx = sorted_idx[-q:].tolist()

    out: List[MiscalNeuron] = []
    for L, A in acts.items():
        a_mean = A.mean(dim=0)
        a_var = ((A - a_mean) ** 2).sum(dim=0).clamp_min(1e-8)
        cov = ((A - a_mean) * (miscal - m_mean).unsqueeze(1)).sum(dim=0)
        r = cov / torch.sqrt(a_var * m_var)
        topk = torch.topk(r, k=min(top_k, r.numel()))
        for r_val, n_idx in zip(topk.values.tolist(), topk.indices.tolist()):
            ni = int(n_idx)
            out.append(MiscalNeuron(
                layer=int(L), neuron=ni,
                pearson_r=float(r_val), n=n,
                mean_act_overconf=float(A[high_idx, ni].mean()),
                mean_act_calib=float(A[low_idx, ni].mean()),
            ))
    out.sort(key=lambda x: x.pearson_r, reverse=True)
    return out[:top_k]
