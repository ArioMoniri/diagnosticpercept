"""H3 — symptom-to-diagnosis routing via residual-stream activation patching.

For each layer L:
  1. Run a *clean* forward (clean prompt → committed clean dx) and cache
     ``layer[L]`` output.
  2. Run a *corrupted* forward (corrupted prompt → committed corrupted dx)
     and replace ``layer[L]`` output with the cached clean tensor.
  3. Measure logit-diff at the last prompt position:
         logit_diff = logits[clean_dx_token] − logits[corrupted_dx_token]
  4. Per-layer score:
         (patched − corrupted) / (clean − corrupted)
     so 0 = no recovery, 1 = full recovery of clean behavior.

Drill: after the per-layer curve identifies the critical layer L*, repeat
patching at the *MLP-neuron* level by replacing only one column of ``h`` in
the corrupted forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from .hooks import ResidualCache, residual_patch
from .model import LoadedModel, clear_h


@dataclass
class PatchScore:
    """One per-layer patching measurement for a single (clean, corrupt) pair."""
    pair_id: str
    layer: int
    clean_diff: float
    corrupt_diff: float
    patched_diff: float
    score: float            # (patched − corrupt) / (clean − corrupt)


def _first_token_id(tokenizer, label: str) -> int:
    ids = tokenizer(" " + label, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Empty tokenization for {label!r}")
    return ids[0]


def _shape_match(cached: torch.Tensor, corrupted_ids: torch.Tensor) -> torch.Tensor:
    """Require identical token counts between clean and corrupted prompts.

    Earlier this function silently padded the clean cache with zeros on the
    prompt side, which placed clean residuals only at the answer-tail
    positions — equivalent to measuring "how well does the answer-tail
    commitment recover," not the routing path. The ROME / IOI convention
    requires token-count-matched (clean, corrupted) pairs so that position-i
    on each run corresponds to the same role token.

    If your H3 pair has mismatched lengths, edit the prompts in
    ``src/data.py`` to swap only the disease-distinguishing words while
    keeping the rest token-identical. Caught by ml-developer review.
    """
    target_len = corrupted_ids.shape[1]
    cur_len = cached.shape[1]
    if cur_len == target_len:
        return cached
    raise ValueError(
        f"H3 patching requires identical-length clean/corrupted pairs; "
        f"got clean_len={cur_len} corrupted_len={target_len}. Rewrite the "
        f"pair in src/data.py H3_PAIRS so token counts match (swap only the "
        f"distinguishing words, keep prompt scaffolding identical)."
    )


@torch.no_grad()
def _logit_diff(
    lm: LoadedModel, input_ids: torch.Tensor, clean_tid: int, corrupt_tid: int
) -> float:
    out = lm.model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, -1].float()
    return float(logits[clean_tid] - logits[corrupt_tid])


@torch.no_grad()
def patch_layers(
    lm: LoadedModel,
    clean_prompt: str,
    corrupt_prompt: str,
    clean_dx: str,
    corrupt_dx: str,
    pair_id: str = "pair",
    layer_indices: Optional[Sequence[int]] = None,
) -> List[PatchScore]:
    """Per-layer residual patching over a single (clean, corrupt) pair."""
    tok = lm.tokenizer
    layer_indices = list(layer_indices) if layer_indices is not None else list(range(lm.n_layers))

    clean_tid = _first_token_id(tok, clean_dx)
    corrupt_tid = _first_token_id(tok, corrupt_dx)

    clean_ids = tok(clean_prompt, return_tensors="pt").input_ids.to(lm.device)
    corrupt_ids = tok(corrupt_prompt, return_tensors="pt").input_ids.to(lm.device)

    # Baselines.
    clean_diff = _logit_diff(lm, clean_ids, clean_tid, corrupt_tid)
    corrupt_diff = _logit_diff(lm, corrupt_ids, clean_tid, corrupt_tid)
    clear_h(lm.layers)

    # Cache clean residuals.
    cache = ResidualCache(lm.layers, layer_indices)
    with cache:
        lm.model(input_ids=clean_ids, use_cache=False)
    clear_h(lm.layers)

    results: List[PatchScore] = []
    denom = clean_diff - corrupt_diff
    for L in tqdm(layer_indices, desc=f"patch {pair_id}", leave=False):
        cached = _shape_match(cache.outputs[L], corrupt_ids)
        with residual_patch(lm.layers, L, cached):
            patched_diff = _logit_diff(lm, corrupt_ids, clean_tid, corrupt_tid)
        clear_h(lm.layers)
        score = (patched_diff - corrupt_diff) / denom if abs(denom) > 1e-6 else 0.0
        results.append(
            PatchScore(
                pair_id=pair_id, layer=L,
                clean_diff=clean_diff, corrupt_diff=corrupt_diff,
                patched_diff=patched_diff, score=float(score),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Neuron-level drill at a critical layer
# ---------------------------------------------------------------------------


@dataclass
class NeuronPatchScore:
    """Per-neuron drill at a fixed critical layer."""
    pair_id: str
    layer: int
    neuron: int
    clean_diff: float
    corrupt_diff: float
    patched_diff: float
    score: float


@torch.no_grad()
def patch_neurons_at_layer(
    lm: LoadedModel,
    clean_prompt: str,
    corrupt_prompt: str,
    clean_dx: str,
    corrupt_dx: str,
    layer_idx: int,
    neuron_indices: Optional[Sequence[int]] = None,
    pair_id: str = "pair",
) -> List[NeuronPatchScore]:
    """Drill: replace ``h[layer_idx, :, neuron]`` from clean run, per neuron.

    If ``neuron_indices`` is None, scan a uniform stride of 64 neurons (smoke-
    test friendly). For real runs, callers should pass a curated subset (e.g.
    H1's top-100 by score) to keep wall-time tractable.
    """
    tok = lm.tokenizer
    clean_tid = _first_token_id(tok, clean_dx)
    corrupt_tid = _first_token_id(tok, corrupt_dx)
    clean_ids = tok(clean_prompt, return_tensors="pt").input_ids.to(lm.device)
    corrupt_ids = tok(corrupt_prompt, return_tensors="pt").input_ids.to(lm.device)

    clean_diff = _logit_diff(lm, clean_ids, clean_tid, corrupt_tid)
    corrupt_diff = _logit_diff(lm, corrupt_ids, clean_tid, corrupt_tid)
    clear_h(lm.layers)

    # Cache clean h at this layer's MLP.
    lm.model(input_ids=clean_ids, use_cache=False)
    clean_h = lm.layers[layer_idx].mlp._h.detach().clone()        # [1, T_clean, d_ff]
    clear_h(lm.layers)

    target_len = corrupt_ids.shape[1]
    if clean_h.shape[1] != target_len:
        if clean_h.shape[1] > target_len:
            clean_h = clean_h[:, -target_len:, :]
        else:
            pad = torch.zeros(
                clean_h.shape[0], target_len - clean_h.shape[1], clean_h.shape[2],
                dtype=clean_h.dtype, device=clean_h.device,
            )
            clean_h = torch.cat([pad, clean_h], dim=1)

    if neuron_indices is None:
        stride = max(1, lm.d_ff // 64)
        neuron_indices = list(range(0, lm.d_ff, stride))

    results: List[NeuronPatchScore] = []
    denom = clean_diff - corrupt_diff

    def make_hook(col: int, vec: torch.Tensor):
        def hook(_m, inputs):
            (h,) = inputs
            h = h.clone()
            # vec is [target_len]; align if corrupted has different runtime length.
            v = vec
            if h.shape[1] != v.shape[0]:
                if h.shape[1] < v.shape[0]:
                    v = v[-h.shape[1]:]
                else:
                    pad = torch.zeros(h.shape[1] - v.shape[0], dtype=v.dtype, device=v.device)
                    v = torch.cat([pad, v], dim=0)
            h[0, :, col] = v.to(h.dtype)
            return (h,)
        return hook

    for n in tqdm(neuron_indices, desc=f"neuron-drill L{layer_idx}", leave=False):
        vec = clean_h[0, :, n]                                    # [T_target]
        handle = lm.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
            make_hook(n, vec)
        )
        try:
            patched_diff = _logit_diff(lm, corrupt_ids, clean_tid, corrupt_tid)
        finally:
            handle.remove()
        clear_h(lm.layers)
        score = (patched_diff - corrupt_diff) / denom if abs(denom) > 1e-6 else 0.0
        results.append(
            NeuronPatchScore(
                pair_id=pair_id, layer=layer_idx, neuron=int(n),
                clean_diff=clean_diff, corrupt_diff=corrupt_diff,
                patched_diff=patched_diff, score=float(score),
            )
        )
    return results
