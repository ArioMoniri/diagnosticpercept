"""H1 — single-neuron diagnosis-gate discovery (paper §2.3).

For neuron ``i`` at layer ``L`` and post-instruction token ``t``:

    score_{i,t} = G_{i,t} · (a^(neg)_{i,t} − a^(pos)_{i,t})       (Eq. 4)
    where G = grad_pos + grad_neg                                (Eq. 3)
    over the log-odds loss  L = -log p_target / (1 - p_target)    (Eq. 2)

Magnitude filter: keep neurons with |a^(pos)| > |a^(neg)| at the winning token.
Search range: first 2/3 of layers (paper convention).
Top-5 reranking sweep: m ∈ {0, ±5, ±10, ±20, ±40, ±80, ±120}.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .hooks import constant_intervention
from .model import LoadedModel, clear_h


# Multiplier sweep for top-5 candidate reranking (paper §2.3 "Reranking").
DEFAULT_M_SWEEP: Tuple[float, ...] = (0.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0,
                                      40.0, -40.0, 80.0, -80.0, 120.0, -120.0)


@dataclass
class NeuronScore:
    """A scored MLP neuron candidate at a specific token position."""
    layer: int
    neuron: int
    token_pos: int
    score: float
    a_pos: float
    a_neg: float
    g_pos: float
    g_neg: float


def _target_first_token_ids(tokenizer, phrases: Sequence[str], icd_tokens: Sequence[str]) -> List[int]:
    """Union of first-token IDs across commitment phrases + ICD codes.

    Phrases prepended with " " so they tokenize as continuations of an
    assistant-style prefix.
    """
    ids = set()
    for s in list(phrases) + list(icd_tokens):
        out = tokenizer(" " + s, add_special_tokens=False).input_ids
        if out:
            ids.add(out[0])
    if not ids:
        raise ValueError("No target token IDs resolved.")
    return sorted(ids)


def _logodds_target_loss(
    logits_last: torch.Tensor, target_ids: Sequence[int]
) -> torch.Tensor:
    """L = -log p_target / (1 - p_target) at the last prompt position (Eq. 2)."""
    probs = F.softmax(logits_last.float(), dim=-1)
    p_target = probs[..., torch.as_tensor(target_ids, device=probs.device)].sum(dim=-1)
    p_target = p_target.clamp(min=1e-12, max=1.0 - 1e-12)
    return -(torch.log(p_target) - torch.log(1.0 - p_target)).mean()


def _per_token_stats(
    lm: LoadedModel,
    prompts: Sequence[str],
    target_ids: Sequence[int],
    layer_indices: Sequence[int],
    max_length: int = 256,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Run forward+backward on each prompt; return per-layer mean a, g over prompts.

    Returns ``{layer_idx: {"a": Tensor[T, d_ff], "g": Tensor[T, d_ff]}}`` where
    ``T`` is the number of post-instruction tokens we monitor (here: the last
    ``WINDOW`` tokens of each prompt — see ``WINDOW`` below).
    """
    WINDOW = 4  # post-instruction tokens to score over (paper monitors a small set T)
    tok = lm.tokenizer
    layers = lm.layers
    device = lm.device

    # Accumulators per layer: sum over prompts of [WINDOW, d_ff] for a and g.
    accum_a: Dict[int, torch.Tensor] = {}
    accum_g: Dict[int, torch.Tensor] = {}
    count = 0

    lm.model.eval()
    for prompt in tqdm(prompts, desc="forward+backward", leave=False):
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        input_ids = enc.input_ids
        attn = enc.attention_mask
        T_total = input_ids.shape[1]
        if T_total < 2:
            continue
        window = min(WINDOW, T_total)

        # Enable grads on the forward pass (model is in eval mode but we need gradients).
        lm.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = lm.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits_last = out.logits[:, -1, :]
            loss = _logodds_target_loss(logits_last, target_ids)
            loss.backward()

        for L in layer_indices:
            mlp = layers[L].mlp
            h = mlp._h                                        # [1, T_total, d_ff]
            g = h.grad if h.grad is not None else torch.zeros_like(h)
            a_win = h[0, -window:, :].detach().float().cpu()   # [window, d_ff]
            g_win = g[0, -window:, :].detach().float().cpu()
            # Pad to WINDOW length so accumulators have a fixed shape.
            if window < WINDOW:
                pad = WINDOW - window
                a_win = F.pad(a_win, (0, 0, pad, 0))
                g_win = F.pad(g_win, (0, 0, pad, 0))
            if L not in accum_a:
                accum_a[L] = torch.zeros_like(a_win)
                accum_g[L] = torch.zeros_like(g_win)
            accum_a[L] += a_win
            accum_g[L] += g_win

        # Free the autograd graph + cached activations between prompts to keep
        # peak VRAM bounded (matters on T4 / L4 where Med42-8B in 4-bit already
        # sits at ~5 GB after load).
        del out, loss
        clear_h(layers)
        lm.model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        count += 1

    if count == 0:
        raise RuntimeError("No prompts processed.")
    return {
        L: {"a": accum_a[L] / count, "g": accum_g[L] / count}
        for L in accum_a
    }


def discover(
    lm: LoadedModel,
    positive_prompts: Sequence[str],
    negative_prompts: Sequence[str],
    target_phrases: Sequence[str],
    icd10_tokens: Sequence[str],
    layer_range: Optional[Tuple[int, int]] = None,
    top_k: int = 5,
) -> List[NeuronScore]:
    """H1 discovery (paper §2.3).

    Parameters
    ----------
    layer_range
        ``(lo, hi)`` decoder layer indices (inclusive lo, exclusive hi). Defaults
        to the first 2/3 of layers per paper.
    top_k
        How many top-scoring neurons to return.
    """
    if layer_range is None:
        hi = max(1, (lm.n_layers * 2) // 3)
        layer_range = (0, hi)
    layer_indices = list(range(*layer_range))

    target_ids = _target_first_token_ids(lm.tokenizer, target_phrases, icd10_tokens)

    stats_pos = _per_token_stats(lm, positive_prompts, target_ids, layer_indices)
    stats_neg = _per_token_stats(lm, negative_prompts, target_ids, layer_indices)

    candidates: List[NeuronScore] = []
    for L in layer_indices:
        a_pos = stats_pos[L]["a"]          # [WINDOW, d_ff]
        a_neg = stats_neg[L]["a"]
        g_pos = stats_pos[L]["g"]
        g_neg = stats_neg[L]["g"]

        G = g_pos + g_neg                                              # Eq. 3
        score = G * (a_neg - a_pos)                                    # Eq. 4

        # Magnitude filter: keep neurons with |a_pos| > |a_neg| (per token).
        mag_mask = a_pos.abs() > a_neg.abs()
        score = score.masked_fill(~mag_mask, float("-inf"))

        # Best token t* per neuron, then top-k neurons at this layer.
        best_t_scores, best_t = score.max(dim=0)                       # [d_ff], [d_ff]
        layer_topk = torch.topk(best_t_scores, k=min(top_k, score.shape[1]))
        for s_val, n_idx in zip(layer_topk.values.tolist(), layer_topk.indices.tolist()):
            if s_val == float("-inf"):
                continue
            t_star = best_t[n_idx].item()
            candidates.append(
                NeuronScore(
                    layer=L,
                    neuron=n_idx,
                    token_pos=int(t_star),
                    score=float(s_val),
                    a_pos=float(a_pos[t_star, n_idx]),
                    a_neg=float(a_neg[t_star, n_idx]),
                    g_pos=float(g_pos[t_star, n_idx]),
                    g_neg=float(g_neg[t_star, n_idx]),
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Multiplier sweep / capability-preserving reranking
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """One (neuron, multiplier) measurement on a held-out probe."""
    layer: int
    neuron: int
    multiplier: float
    target_logprob: float       # mean log p(target | prompt) over probes (lower = more suppression)
    sample_generation: str


@torch.no_grad()
def _mean_target_logprob(
    lm: LoadedModel,
    prompts: Sequence[str],
    target_ids: Sequence[int],
    max_length: int = 256,
) -> float:
    """Mean log P(target ∈ target_ids) at the last prompt token across prompts."""
    tok = lm.tokenizer
    total = 0.0
    n = 0
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=max_length).to(lm.device)
        out = lm.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        logp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
        p_target = torch.logsumexp(
            logp[torch.as_tensor(target_ids, device=logp.device)], dim=0
        )
        total += float(p_target)
        n += 1
        clear_h(lm.layers)
    return total / max(1, n)


@torch.no_grad()
def _sample_one(lm: LoadedModel, prompt: str, max_new_tokens: int = 48) -> str:
    """Quick greedy generation for inline inspection."""
    enc = lm.tokenizer(prompt, return_tensors="pt").to(lm.device)
    out_ids = lm.model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=lm.tokenizer.pad_token_id,
    )
    clear_h(lm.layers)
    return lm.tokenizer.decode(out_ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def sweep(
    lm: LoadedModel,
    candidates: Sequence[NeuronScore],
    probes: Sequence[str],
    target_phrases: Sequence[str],
    icd10_tokens: Sequence[str],
    multipliers: Sequence[float] = DEFAULT_M_SWEEP,
    sample_prompt: Optional[str] = None,
) -> List[SweepResult]:
    """Sweep ``m`` for each candidate; report mean target log-prob on probes.

    Lower target log-prob under the intervention than baseline (m=0) indicates
    the neuron is doing real work toward the diagnosis-commitment signal — the
    direction we care about for H1.
    """
    target_ids = _target_first_token_ids(lm.tokenizer, target_phrases, icd10_tokens)
    out: List[SweepResult] = []
    for cand in tqdm(candidates, desc="sweep", leave=False):
        for m in multipliers:
            with constant_intervention(lm.layers, cand.neuron, m, cand.layer):
                lp = _mean_target_logprob(lm, probes, target_ids)
                gen = _sample_one(lm, sample_prompt) if sample_prompt else ""
            out.append(
                SweepResult(
                    layer=cand.layer,
                    neuron=cand.neuron,
                    multiplier=float(m),
                    target_logprob=lp,
                    sample_generation=gen,
                )
            )
    return out


def best_multiplier(sweep_results: Sequence[SweepResult]) -> Tuple[int, int, float]:
    """Pick the (layer, neuron, m) that minimizes target log-prob (max suppression)."""
    if not sweep_results:
        raise ValueError("Empty sweep results.")
    best = min(sweep_results, key=lambda r: r.target_logprob)
    return best.layer, best.neuron, best.multiplier
