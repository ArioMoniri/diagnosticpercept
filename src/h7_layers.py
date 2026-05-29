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

        # Re-run a forward over prompt + tokens UP TO (but not including) the
        # answer-letter token. The last position of h on that forward
        # corresponds to the logit that *produced* the answer letter — exactly
        # the activation we want to correlate with miscalibration. The previous
        # version went one token too far (h[-1] then was the position whose
        # logit predicts the token AFTER the letter). Caught by ml-developer
        # review 2026-05-29.
        full_ids = torch.cat([enc.input_ids[0], generated_ids[:ans_pos]]).unsqueeze(0)
        lm.model(input_ids=full_ids, use_cache=False)
        for L in layer_indices:
            h = lm.layers[L].mlp._h.detach().float()  # [1, T, d_ff]
            # The answer-position activation IS at the last position now.
            ans_act = h[0, -1, :].cpu()  # [d_ff]
            acts_per_layer[L].append(ans_act)
        clear_h(lm.layers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows.append({
            "q_id": item.q_id, "gold": item.gold,
            "predicted": predicted_letter,
            "p_top1_at_answer": p_top1,
            "p_gold_at_answer": p_gold,
            "correct": correct,
            # M7 (ml-developer review 2026-05-29): reasoning-chain length is a
            # known confound for calibration — longer chains are correlated
            # with hedging, which lowers p_top1. We record it here so the
            # ranker can stratify by length quartile (see
            # `rank_miscalibration_neurons(..., length_binned=True)`).
            "chain_len": int(ans_pos),
        })

    stacked = {L: torch.stack(acts_per_layer[L], dim=0) for L in layer_indices if acts_per_layer[L]}
    return rows, stacked


def _pearson_r_pvalue(r: torch.Tensor, n: int) -> torch.Tensor:
    """Two-sided p-value for Pearson r at sample size n. Vectorized."""
    # t = r * sqrt(n-2) / sqrt(1 - r^2)
    r = r.clamp(-0.9999, 0.9999)
    t = r * torch.sqrt(torch.tensor(n - 2, dtype=r.dtype)) / torch.sqrt(1 - r ** 2)
    # Approximate two-sided p via complementary error of student-t. For large
    # n the Student-t converges to normal; use the normal approximation
    # (n >> 30 in all our use cases). We import scipy lazily to keep the
    # module light.
    try:
        from scipy import stats
        return torch.tensor(
            2 * (1 - stats.t.cdf(t.abs().numpy(), df=n - 2)), dtype=r.dtype
        )
    except ImportError:
        # Normal-approximation fallback (no scipy required).
        import math
        # erfc(|t|/sqrt(2)) — survival function of |Z|; doubled for two-sided.
        return torch.tensor(
            [math.erfc(float(abs(ti)) / math.sqrt(2)) for ti in t],
            dtype=r.dtype,
        )


def _bh_fdr(p: torch.Tensor, q: float = 0.05) -> torch.Tensor:
    """Benjamini-Hochberg FDR: return Boolean mask of p-values to reject."""
    n = p.numel()
    order = torch.argsort(p)
    p_sorted = p[order]
    thresh = torch.arange(1, n + 1, dtype=p.dtype) * q / n
    pass_mask = p_sorted <= thresh
    if not pass_mask.any():
        return torch.zeros_like(p, dtype=torch.bool)
    k = int(pass_mask.nonzero().max().item()) + 1
    cutoff = p_sorted[k - 1]
    return p <= cutoff


def rank_miscalibration_neurons(
    rows: Sequence[Dict[str, float]],
    acts: Dict[int, torch.Tensor],
    top_k: int = 20,
    fdr_q: float = 0.05,
    length_binned: bool = False,
) -> List[MiscalNeuron]:
    """Per-neuron Pearson r between activation and ``miscal = p_top1 − correct``.

    Applies Benjamini-Hochberg FDR (M2 from review): with ~13k neurons × 16
    layers ≈ 200K tests, top-k by raw r contained many false positives at
    realistic q. We BH-correct over the full neuron × layer pool and only
    keep neurons whose FDR-adjusted p ≤ ``fdr_q``, then take top-k by r.

    ``length_binned`` (M7): if True, partition rows into reasoning-chain-length
    quartiles (using the ``chain_len`` field recorded by
    :func:`collect_answer_position_acts`) and rank within each bin. The reported
    Pearson r is then the *mean across bins* — a neuron whose r survives length
    stratification reflects calibration, not chain-length effects. Returns an
    empty list if any bin has < 3 cases (Pearson undefined).
    """
    if length_binned:
        return _rank_length_binned(rows, acts, top_k=top_k, fdr_q=fdr_q)
    miscal = torch.tensor(
        [r["p_top1_at_answer"] - float(r["correct"]) for r in rows], dtype=torch.float32
    )  # [N]
    m_mean = miscal.mean()
    m_var = ((miscal - m_mean) ** 2).sum().clamp_min(1e-8)
    n = len(miscal)

    # Subset definitions for diagnostic mean-act fields.
    q_count = max(1, n // 4)
    sorted_idx = torch.argsort(miscal, descending=True)
    high_idx = sorted_idx[:q_count].tolist()
    low_idx = sorted_idx[-q_count:].tolist()

    # Compute r for every (layer, neuron); then BH-correct globally.
    all_r: List[float] = []
    all_layer: List[int] = []
    all_neuron: List[int] = []
    all_A: Dict[int, torch.Tensor] = {}
    for L, A in acts.items():
        a_mean = A.mean(dim=0)
        a_var = ((A - a_mean) ** 2).sum(dim=0).clamp_min(1e-8)
        cov = ((A - a_mean) * (miscal - m_mean).unsqueeze(1)).sum(dim=0)
        r = cov / torch.sqrt(a_var * m_var)
        for n_idx in range(r.numel()):
            all_r.append(float(r[n_idx]))
            all_layer.append(int(L))
            all_neuron.append(int(n_idx))
        all_A[int(L)] = A

    r_tensor = torch.tensor(all_r)
    p_tensor = _pearson_r_pvalue(r_tensor, n)
    pass_mask = _bh_fdr(p_tensor, q=fdr_q)
    print(f"H7 FDR-corrected: {int(pass_mask.sum())}/{len(all_r)} neurons "
          f"pass q={fdr_q} (would-be top-k by raw r had no correction)")

    out: List[MiscalNeuron] = []
    for i, (L, ni, r_val) in enumerate(zip(all_layer, all_neuron, all_r)):
        if not bool(pass_mask[i].item()):
            continue
        out.append(MiscalNeuron(
            layer=L, neuron=ni, pearson_r=r_val, n=n,
            mean_act_overconf=float(all_A[L][high_idx, ni].mean()),
            mean_act_calib=float(all_A[L][low_idx, ni].mean()),
        ))
    out.sort(key=lambda x: x.pearson_r, reverse=True)
    return out[:top_k]


def _rank_length_binned(
    rows: Sequence[Dict[str, float]],
    acts: Dict[int, torch.Tensor],
    top_k: int = 20,
    fdr_q: float = 0.05,
    n_bins: int = 4,
) -> List[MiscalNeuron]:
    """M7: Pearson r averaged across length-stratified bins.

    Within-bin Pearson removes the additive effect of chain length on both
    activation and miscalibration. A neuron whose mean r across bins survives
    BH-FDR over the union of all (layer × neuron × bin) tests is reported.
    Robust to the case where ``chain_len`` is missing on some rows (those rows
    are dropped before binning).
    """
    rows_with_len = [
        (i, r) for i, r in enumerate(rows) if "chain_len" in r and r["chain_len"] is not None
    ]
    if len(rows_with_len) < n_bins * 3:
        print(f"[H7 length-binned] only {len(rows_with_len)} rows with chain_len; "
              f"need >= {n_bins * 3} for {n_bins}-bin analysis. Falling back to raw.")
        return rank_miscalibration_neurons(rows, acts, top_k=top_k, fdr_q=fdr_q,
                                          length_binned=False)

    lens = torch.tensor([r["chain_len"] for _, r in rows_with_len], dtype=torch.float32)
    # Quantile cutoffs.
    qs = torch.linspace(0, 1, n_bins + 1)[1:-1]
    cuts = torch.quantile(lens, qs)
    # Bin index per row.
    bin_idx = torch.zeros(len(lens), dtype=torch.long)
    for cut in cuts:
        bin_idx += (lens > cut).long()

    # Per-bin Pearson r per neuron per layer.
    bin_rs: Dict[int, Dict[int, List[float]]] = {}  # layer -> neuron -> [r_bin0, r_bin1, ...]
    for b in range(n_bins):
        b_mask = (bin_idx == b)
        b_rows_idx = [rows_with_len[i][0] for i in range(len(rows_with_len)) if b_mask[i]]
        if len(b_rows_idx) < 3:
            continue
        miscal_b = torch.tensor(
            [rows[i]["p_top1_at_answer"] - float(rows[i]["correct"]) for i in b_rows_idx],
            dtype=torch.float32,
        )
        m_mean = miscal_b.mean()
        m_var = ((miscal_b - m_mean) ** 2).sum().clamp_min(1e-8)
        for L, A_full in acts.items():
            A = A_full[b_rows_idx]
            a_mean = A.mean(dim=0)
            a_var = ((A - a_mean) ** 2).sum(dim=0).clamp_min(1e-8)
            cov = ((A - a_mean) * (miscal_b - m_mean).unsqueeze(1)).sum(dim=0)
            r = cov / torch.sqrt(a_var * m_var)
            bin_rs.setdefault(L, {})
            for n_idx in range(r.numel()):
                bin_rs[L].setdefault(n_idx, []).append(float(r[n_idx]))

    # Mean r per (layer, neuron) over bins; require all bins contributed.
    all_layer: List[int] = []
    all_neuron: List[int] = []
    all_r: List[float] = []
    for L, ndict in bin_rs.items():
        for n_idx, rs in ndict.items():
            if len(rs) < n_bins:
                continue
            all_layer.append(int(L))
            all_neuron.append(int(n_idx))
            all_r.append(float(sum(rs) / len(rs)))

    if not all_r:
        return []

    # Treat each (L, neuron) as a single Pearson r at the smallest bin's N.
    n_min = min(
        int((bin_idx == b).sum().item())
        for b in range(n_bins)
        if int((bin_idx == b).sum().item()) >= 3
    )
    r_tensor = torch.tensor(all_r)
    p_tensor = _pearson_r_pvalue(r_tensor, n_min)
    pass_mask = _bh_fdr(p_tensor, q=fdr_q)
    print(f"H7 length-binned + FDR: {int(pass_mask.sum())}/{len(all_r)} neurons "
          f"pass q={fdr_q} after {n_bins}-bin length stratification (n_min={n_min})")

    n_total = len(rows_with_len)
    # Diagnostic mean acts use overall (not within-bin) high/low miscal subsets.
    miscal_full = torch.tensor(
        [rows[i]["p_top1_at_answer"] - float(rows[i]["correct"]) for i, _ in rows_with_len],
        dtype=torch.float32,
    )
    q_count = max(1, n_total // 4)
    sorted_idx = torch.argsort(miscal_full, descending=True)
    high_idx = [rows_with_len[i][0] for i in sorted_idx[:q_count].tolist()]
    low_idx = [rows_with_len[i][0] for i in sorted_idx[-q_count:].tolist()]

    out: List[MiscalNeuron] = []
    for i, (L, ni, r_val) in enumerate(zip(all_layer, all_neuron, all_r)):
        if not bool(pass_mask[i].item()):
            continue
        out.append(MiscalNeuron(
            layer=L, neuron=ni, pearson_r=r_val, n=n_total,
            mean_act_overconf=float(acts[L][high_idx, ni].mean()),
            mean_act_calib=float(acts[L][low_idx, ni].mean()),
        ))
    out.sort(key=lambda x: x.pearson_r, reverse=True)
    return out[:top_k]
