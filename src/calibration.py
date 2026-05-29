"""H5 — Overconfidence / miscalibration neurons.

The signal we're after:

  *The model's actual probability of the diagnosis token is LOW
   (it doesn't really know), but when asked to attest, it says HIGH
   confidence ("Yes, I am confident").*

Pipeline per case (a hard diagnostic vignette):

  1. ``prompt_dx``   = "<case>\nDiagnosis:"
  2. Greedy-decode 1 token; record:
        dx_token, p_dx = max_prob over the diagnosis distribution
  3. ``prompt_attest`` = "<case>\nDiagnosis: <dx_text>\nAre you confident in
       this diagnosis? Answer yes or no.\nAnswer:"
  4. Get the next-token distribution; record:
        p_yes, p_no over the confident/unsure token sets
  5. **calibration_gap = p_yes − p_dx**
        large positive → high stated confidence on a low-probability dx
        (model is over-confident / miscalibrated)
  6. Capture per-layer signed-max activations during the *attest* forward.
  7. Rank neurons by their correlation (across cases) between their
     activation and the calibration_gap. High positive correlation → neuron
     fires harder when the model is *more* overconfident than warranted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .model import LoadedModel, clear_h


# Tokens we treat as expressing confidence vs hedging on the attest prompt.
CONFIDENT_TOKENS: Tuple[str, ...] = (
    "Yes", "yes", " Yes", " yes",
    "Sure", " Sure", " Definitely", " Absolutely", "Confident", " Confident",
)
UNSURE_TOKENS: Tuple[str, ...] = (
    "No", "no", " No", " no", " Maybe", " Unsure", " Possibly",
)


@dataclass
class CalibrationCase:
    """One hard vignette's calibration measurement."""

    case: str
    dx_text: str
    p_dx: float                          # model's probability of its own dx token
    p_yes: float                         # probability of confident attestation
    p_no: float                          # probability of hedge attestation
    calibration_gap: float               # p_yes - p_dx (overconfidence signal)
    attest_prompt: str
    # Per-layer signed-max activations during the attest forward, [d_ff] each.
    attest_acts: Dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class OverconfidenceNeuron:
    """Neuron whose activation predicts overconfidence."""

    layer: int
    neuron: int
    pearson_r: float                     # corr(activation, calibration_gap)
    n_cases: int
    mean_act_overconf: float             # mean activation on high-gap cases
    mean_act_calib: float                # mean activation on low-gap cases


def _gather_token_ids(tokenizer, strs: Sequence[str]) -> List[int]:
    """First-token IDs for each string (may differ for "Yes" vs " Yes")."""
    ids: List[int] = []
    for s in strs:
        out = tokenizer(s, add_special_tokens=False).input_ids
        if out:
            ids.append(out[0])
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# Per-case measurement
# ---------------------------------------------------------------------------


@torch.no_grad()
def measure_case(
    lm: LoadedModel,
    case: str,
    layer_indices: Sequence[int],
    dx_max_new_tokens: int = 8,
    yes_tokens: Optional[Sequence[int]] = None,
    no_tokens: Optional[Sequence[int]] = None,
) -> CalibrationCase:
    """Run dx + attest forwards for one case; capture probabilities and acts.

    Notes on token probability semantics:
      ``p_dx`` is the max softmax probability over the next-token distribution
      conditioned on the dx prompt — i.e. ``max_t softmax(logits)[t]``. This is
      the model's actual top-1 confidence in *some* token; if it is low, the
      model is genuinely uncertain about what to say next. We then decode the
      greedy chain to recover ``dx_text`` for the attestation prompt.
    """
    tok = lm.tokenizer
    if yes_tokens is None:
        yes_tokens = _gather_token_ids(tok, CONFIDENT_TOKENS)
    if no_tokens is None:
        no_tokens = _gather_token_ids(tok, UNSURE_TOKENS)

    # ---- Step 1-2: diagnosis prompt + decode + p_dx ----
    dx_prompt = f"{case}\nDiagnosis:"
    enc = tok(dx_prompt, return_tensors="pt").to(lm.device)
    gen = lm.model.generate(
        **enc, max_new_tokens=dx_max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id,
        output_scores=True, return_dict_in_generate=True,
    )
    clear_h(lm.layers)

    # First-token probability = model's confidence in its top choice for the dx slot.
    first_probs = F.softmax(gen.scores[0][0].float(), dim=-1)
    p_dx = float(first_probs.max())

    dx_ids = gen.sequences[0, enc.input_ids.shape[1]:]
    dx_text = tok.decode(dx_ids, skip_special_tokens=True).strip()

    # ---- Step 3-4: attestation prompt ----
    attest_prompt = (
        f"{case}\nDiagnosis: {dx_text}\n"
        f"Are you confident in this diagnosis? Answer yes or no.\nAnswer:"
    )
    enc2 = tok(attest_prompt, return_tensors="pt").to(lm.device)
    out2 = lm.model(input_ids=enc2.input_ids, use_cache=False)
    last_logits = out2.logits[0, -1].float()
    probs = F.softmax(last_logits, dim=-1)
    p_yes = float(probs[torch.as_tensor(yes_tokens, device=probs.device)].sum())
    p_no = float(probs[torch.as_tensor(no_tokens, device=probs.device)].sum())

    # ---- Step 6: capture per-layer signed-max activations during attest forward ----
    acts: Dict[int, torch.Tensor] = {}
    for L in layer_indices:
        h = lm.layers[L].mlp._h.detach().float()       # [1, T, d_ff]
        argmax = h.abs().argmax(dim=1, keepdim=True)
        signed = h.gather(1, argmax).squeeze(1).squeeze(0).cpu()
        acts[L] = signed
    clear_h(lm.layers)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return CalibrationCase(
        case=case,
        dx_text=dx_text,
        p_dx=p_dx,
        p_yes=p_yes,
        p_no=p_no,
        calibration_gap=p_yes - p_dx,
        attest_prompt=attest_prompt,
        attest_acts=acts,
    )


# ---------------------------------------------------------------------------
# Pipeline + ranking
# ---------------------------------------------------------------------------


def find_overconfidence_neurons(
    lm: LoadedModel,
    hard_cases: Sequence[str],
    layer_range: Optional[Tuple[int, int]] = None,
    top_k: int = 10,
    overconf_threshold: float = 0.3,
    gap_high_low_n: int = 4,
) -> Tuple[List[CalibrationCase], List[OverconfidenceNeuron]]:
    """End-to-end H5.

    Parameters
    ----------
    hard_cases
        Diagnostic vignettes the unmodified model has to reason about.
    layer_range
        ``(lo, hi)``; default = later half (commitment/confidence is closer
        to the head).
    top_k
        How many overconfidence neurons to return.
    overconf_threshold
        A case is *overconfident* if calibration_gap >= this. Used only for
        the high/low subset means; the Pearson correlation uses all cases.
    gap_high_low_n
        If we want non-trivial subset means, take the top/bottom-n cases by
        calibration_gap.
    """
    if layer_range is None:
        layer_range = (lm.n_layers // 2, lm.n_layers)
    layer_indices = list(range(*layer_range))

    cases: List[CalibrationCase] = []
    for case in tqdm(hard_cases, desc="H5 measure", leave=False):
        cases.append(measure_case(lm, case, layer_indices))

    if len(cases) < 3:
        raise RuntimeError("Need >=3 cases to compute correlations.")

    gaps = torch.tensor([c.calibration_gap for c in cases], dtype=torch.float32)  # [N]
    g_mean = gaps.mean()
    g_var = ((gaps - g_mean) ** 2).sum().clamp_min(1e-8)

    # Sort cases by gap to define high/low subsets.
    sorted_idx = torch.argsort(gaps, descending=True)
    high_idx = sorted_idx[:gap_high_low_n].tolist()
    low_idx = sorted_idx[-gap_high_low_n:].tolist()

    ranked: List[OverconfidenceNeuron] = []
    for L in layer_indices:
        # Stack [N, d_ff].
        A = torch.stack([c.attest_acts[L] for c in cases], dim=0)
        a_mean = A.mean(dim=0)
        a_var = ((A - a_mean) ** 2).sum(dim=0).clamp_min(1e-8)
        cov = ((A - a_mean) * (gaps - g_mean).unsqueeze(1)).sum(dim=0)
        r = cov / torch.sqrt(a_var * g_var)
        # Want neurons whose activation *positively* predicts overconfidence.
        topk = torch.topk(r, k=min(top_k, r.numel()))
        for r_val, n_idx in zip(topk.values.tolist(), topk.indices.tolist()):
            n = int(n_idx)
            ranked.append(
                OverconfidenceNeuron(
                    layer=L, neuron=n,
                    pearson_r=float(r_val),
                    n_cases=len(cases),
                    mean_act_overconf=float(A[high_idx, n].mean()),
                    mean_act_calib=float(A[low_idx, n].mean()),
                )
            )

    ranked.sort(key=lambda x: x.pearson_r, reverse=True)
    return cases, ranked[:top_k]


# ---------------------------------------------------------------------------
# H5 at scale (M1): route through H7's MCQ-style collector for N >= 200
# ---------------------------------------------------------------------------


def find_overconfidence_neurons_at_scale(
    lm: LoadedModel,
    items: Sequence,
    layer_range: Optional[Tuple[int, int]] = None,
    top_k: int = 20,
    fdr_q: float = 0.05,
    max_new_tokens: int = 220,
    length_binned: bool = True,
):
    """H5 at MedQA scale (N ≥ 200) via the H7 calibration signal.

    The original H5 (``find_overconfidence_neurons``) measures
    ``p_yes − p_dx`` on free-form vignettes — well-defined, but tiny N (~20 hard
    cases) gives noisy Pearson rs and no way to FDR-correct over 13k×16
    neurons. M1 (ml-developer review 2026-05-29) re-runs the H5 *correlation*
    over the H7 MCQ-scale signal ``miscal = p_top1 − int(correct)`` for the
    MedQA test set, where N=500+ makes even r≈0.1 detectable with BH-FDR.

    The two H5 variants measure related but different overconfidence axes:

      * H5 free-form: confidence on the *attestation* prompt minus actual top-1
        probability on the *open-ended* dx prompt — captures "I sound confident
        but I'm guessing".
      * H5-at-scale (this fn): confidence at the answer letter minus correctness
        — captures "I committed to a wrong letter with high p_top1".

    Returns the same ``MiscalNeuron`` type as :func:`h7_layers.rank_miscalibration_neurons`.
    """
    from .h7_layers import collect_answer_position_acts, rank_miscalibration_neurons

    if layer_range is None:
        layer_indices = list(range(lm.n_layers // 2, lm.n_layers))
    else:
        layer_indices = list(range(*layer_range))

    rows, acts = collect_answer_position_acts(
        lm, items, layer_indices=layer_indices, max_new_tokens=max_new_tokens,
    )
    if len(rows) < 30:
        raise RuntimeError(
            f"H5-at-scale: only {len(rows)} questions yielded a parseable "
            "answer position — model likely isn't honoring the prompt format."
        )
    # iter-6: forward `length_binned` so H5-at-scale gets M7 stratification by
    # default — without this, chain-length confounds bleed into the Pearson r.
    return rows, rank_miscalibration_neurons(
        rows, acts, top_k=top_k, fdr_q=fdr_q, length_binned=length_binned,
    )
