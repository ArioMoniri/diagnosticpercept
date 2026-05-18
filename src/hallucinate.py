"""H4 — Hallucination / false-confidence neurons.

The intuition: H1's gradient×activation picks up neurons that drive the
*commitment* signal — phrases like 'The diagnosis is X', 'I21', etc.
But commitment alone doesn't separate correct diagnosis from confabulation.
A clinician confronted with the *trap* set (under-specified, contradictory,
or fabricated cases) refuses or asks for more data. The model usually commits
anyway and produces a confident hallucination.

The neurons that fire on **trap-committed** prompts but stay silent on
**ambiguous (correctly hedged)** prompts identify *false-confidence* circuitry.
Subtracting H1's pathognomonic-committed activation map isolates the
hallucination-specific component:

    hallucination_score_i = a_i(trap_commit) - a_i(pathognomonic_commit)

i.e. neurons firing more on confabulated commitment than on real one.

A small positive score = neuron fires equally on both = part of the general
gate. A large positive score = neuron fires harder on trap = *false confidence*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .discover import _target_first_token_ids
from .model import LoadedModel, clear_h


@dataclass
class HallucinationNeuron:
    """A neuron whose activation pattern flags hallucination-style commitment."""

    layer: int
    neuron: int
    a_trap: float           # mean signed-max activation on trap-committed prompts
    a_pathog: float         # mean activation on pathognomonic-committed prompts
    a_hedge: float          # mean activation on ambiguous/hedged prompts
    delta: float            # a_trap - a_pathog (hallucination-specific signal)
    commit_gap: float       # a_trap - a_hedge (gate-like commitment signal)


# ---------------------------------------------------------------------------
# Commitment detection: did the model commit to a diagnosis?
# ---------------------------------------------------------------------------


@torch.no_grad()
def did_commit(
    lm: LoadedModel,
    prompt: str,
    target_ids: Sequence[int],
    p_threshold: float = 0.10,
    max_new_tokens: int = 16,
) -> Tuple[bool, str]:
    """Greedy-decode ``max_new_tokens`` from ``prompt`` and check whether the
    first generated token has probability mass > ``p_threshold`` over the
    commitment-token set (or whether any of the first few tokens is one of them).

    Returns ``(committed, generated_text)``.
    """
    tok = lm.tokenizer
    enc = tok(prompt, return_tensors="pt").to(lm.device)
    out = lm.model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True,
    )
    clear_h(lm.layers)

    first_logits = out.scores[0][0].float()
    probs = F.softmax(first_logits, dim=-1)
    target = torch.as_tensor(list(target_ids), device=probs.device)
    p_commit = float(probs[target].sum())

    new_ids = out.sequences[0, enc.input_ids.shape[1]:]
    new_ids_list = new_ids.tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    has_target = any(tid in new_ids_list[:6] for tid in target_ids)
    committed = (p_commit > p_threshold) or has_target
    return committed, text


# ---------------------------------------------------------------------------
# Per-bucket activation aggregation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _signed_max_activations(
    lm: LoadedModel,
    prompts: Sequence[str],
    layer_indices: Sequence[int],
    max_length: int = 256,
) -> Dict[int, torch.Tensor]:
    """For each prompt, the value of ``h`` at the token where ``|h|`` is largest.

    Returns ``{layer: Tensor[n_prompts, d_ff]}`` on CPU.
    """
    out: Dict[int, List[torch.Tensor]] = {L: [] for L in layer_indices}
    tok = lm.tokenizer
    for p in tqdm(prompts, desc="H4 acts", leave=False):
        enc = tok(p, return_tensors="pt", truncation=True, max_length=max_length).to(lm.device)
        lm.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        for L in layer_indices:
            h = lm.layers[L].mlp._h.detach().float()
            argmax = h.abs().argmax(dim=1, keepdim=True)
            signed = h.gather(1, argmax).squeeze(1).squeeze(0).cpu()
            out[L].append(signed)
        clear_h(lm.layers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {L: torch.stack(out[L], dim=0) for L in layer_indices}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def find_hallucination_neurons(
    lm: LoadedModel,
    trap_prompts: Sequence[str],
    pathognomonic_prompts: Sequence[str],
    hedge_prompts: Sequence[str],
    target_phrases: Sequence[str],
    icd10_tokens: Sequence[str],
    layer_range: Optional[Tuple[int, int]] = None,
    top_k: int = 10,
    commit_p_threshold: float = 0.10,
) -> Tuple[List[HallucinationNeuron], Dict[str, List[Tuple[str, bool, str]]]]:
    """Identify neurons whose firing flags *false-confidence* commitment.

    Workflow:
      1. Sample greedy generations from each prompt; classify commit / hedge.
      2. Restrict the *trap* set to prompts the model actually committed to
         (skip the cases where it correctly refused or hedged — they're not
         hallucinations).
      3. Aggregate per-layer signed-max activations on each bucket.
      4. For each neuron compute:
            delta      = mean_trap_commit  - mean_pathog
            commit_gap = mean_trap_commit  - mean_hedge
      5. Rank by ``delta`` (positive = fires harder on hallucinated commitment).

    Returns ``(top_neurons, classifications)`` where ``classifications`` maps
    each bucket to ``[(prompt, committed, generation), ...]``.
    """
    if layer_range is None:
        # Search later layers — commitment is closer to the head.
        lo = max(1, lm.n_layers // 3)
        layer_range = (lo, lm.n_layers)
    layer_indices = list(range(*layer_range))

    target_ids = _target_first_token_ids(lm.tokenizer, target_phrases, icd10_tokens)

    # --- classify each prompt ---
    classifications: Dict[str, List[Tuple[str, bool, str]]] = {}
    for label, prompts in [
        ("trap", trap_prompts),
        ("pathognomonic", pathognomonic_prompts),
        ("hedge", hedge_prompts),
    ]:
        classifications[label] = []
        for p in tqdm(prompts, desc=f"classify {label}", leave=False):
            committed, gen = did_commit(lm, p, target_ids, p_threshold=commit_p_threshold)
            classifications[label].append((p, committed, gen))

    trap_committed = [p for p, c, _ in classifications["trap"] if c]
    pathog_committed = [p for p, c, _ in classifications["pathognomonic"] if c]
    hedge_committed = [p for p, c, _ in classifications["hedge"] if not c]
    if not trap_committed:
        raise RuntimeError(
            "Model never committed on any trap prompt — adjust commit_p_threshold "
            "or the trap set."
        )
    if not pathog_committed:
        raise RuntimeError(
            "Model never committed on pathognomonic prompts — likely a target-"
            "token mismatch."
        )

    # --- activations per bucket ---
    a_trap = _signed_max_activations(lm, trap_committed, layer_indices)
    a_pathog = _signed_max_activations(lm, pathog_committed, layer_indices)
    a_hedge = (
        _signed_max_activations(lm, hedge_committed, layer_indices)
        if hedge_committed else
        {L: torch.zeros(1, lm.d_ff) for L in layer_indices}
    )

    # --- rank ---
    scored: List[HallucinationNeuron] = []
    for L in layer_indices:
        m_trap = a_trap[L].mean(0)
        m_pathog = a_pathog[L].mean(0)
        m_hedge = a_hedge[L].mean(0)
        delta = m_trap - m_pathog
        commit_gap = m_trap - m_hedge

        # Require neuron to fire positively on commitment (gate-like) AND
        # fire harder on hallucinated commitment than on real one.
        keep = (m_trap.abs() > m_hedge.abs() + 1e-3) & (delta.abs() > 0.05)
        d_masked = delta.masked_fill(~keep, float("-inf"))
        topk = torch.topk(d_masked, k=min(top_k, d_masked.numel()))
        for s_val, n_idx in zip(topk.values.tolist(), topk.indices.tolist()):
            if s_val == float("-inf"):
                continue
            scored.append(
                HallucinationNeuron(
                    layer=L, neuron=int(n_idx),
                    a_trap=float(m_trap[n_idx]),
                    a_pathog=float(m_pathog[n_idx]),
                    a_hedge=float(m_hedge[n_idx]),
                    delta=float(delta[n_idx]),
                    commit_gap=float(commit_gap[n_idx]),
                )
            )
    scored.sort(key=lambda x: x.delta, reverse=True)
    return scored[:top_k], classifications
