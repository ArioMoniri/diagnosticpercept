"""Intervention context managers.

Three interventions, all implemented as forward pre-hooks on ``mlp.down_proj``
so they modify ``h`` *after* it is computed but *before* it is projected back
to the residual stream.

- :func:`constant_intervention` — Eq. 5: ``h_i ← m`` at every token.
- :func:`additive_intervention` — paper §4: ``h_i ← h_i + m`` (concept amplification).
- :func:`anchor_intervention`    — Eq. 7: ``h_i ← clamp(k·m·v/d, m)`` per token.
- :func:`residual_patch`         — H3: replace a decoder layer's output with a
  cached tensor from a clean forward (Meng 2022 ROME style).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLP-neuron interventions (forward pre-hooks on down_proj)
# ---------------------------------------------------------------------------


def _make_constant_hook(neuron_idx: int, value: float):
    def hook(_module, inputs):
        (h,) = inputs
        h = h.clone()
        h[..., neuron_idx] = value
        return (h,)

    return hook


def _make_additive_hook(neuron_idx: int, amount: float):
    def hook(_module, inputs):
        (h,) = inputs
        h = h.clone()
        h[..., neuron_idx] = h[..., neuron_idx] + amount
        return (h,)

    return hook


def _make_anchor_hook(
    neuron_idx: int, m_star: float, d: float, k: float = 1.0
) -> "callable":
    """Eq. 7: ``h_i ← clamp(k · m* · v_t / d, m*)`` with ``v_t = h_i[t]``.

    Sign of ``m*`` controls the clamp direction:
      * ``m* > 0``  → ``min(value, m*)``
      * ``m* < 0``  → ``max(value, m*)``
    """
    if d == 0:
        raise ValueError("anchor intervention: d (activation gap) is zero.")

    def hook(_module, inputs):
        (h,) = inputs
        v = h[..., neuron_idx]  # per-token natural activation
        scaled = k * m_star * v / d
        if m_star > 0:
            new = torch.clamp(scaled, max=m_star)
        else:
            new = torch.clamp(scaled, min=m_star)
        h = h.clone()
        h[..., neuron_idx] = new
        return (h,)

    return hook


@contextmanager
def constant_intervention(
    layers: Sequence[nn.Module], neuron_idx: int, value: float, layer_idx: int
) -> Iterator[None]:
    """Pin ``h[layer_idx, neuron_idx] ← value`` for the duration of the block (Eq. 5)."""
    h = layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
        _make_constant_hook(neuron_idx, value)
    )
    try:
        yield
    finally:
        h.remove()


@contextmanager
def additive_intervention(
    layers: Sequence[nn.Module], neuron_idx: int, amount: float, layer_idx: int
) -> Iterator[None]:
    """Add ``amount`` to ``h[layer_idx, neuron_idx]`` (concept amplification, §4)."""
    h = layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
        _make_additive_hook(neuron_idx, amount)
    )
    try:
        yield
    finally:
        h.remove()


@contextmanager
def anchor_intervention(
    layers: Sequence[nn.Module],
    neuron_idx: int,
    m_star: float,
    d: float,
    layer_idx: int,
    k: float = 1.0,
) -> Iterator[None]:
    """Eq. 7 anchor variant: capability-preserving, per-token-scaled."""
    h = layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
        _make_anchor_hook(neuron_idx, m_star, d, k)
    )
    try:
        yield
    finally:
        h.remove()


# ---------------------------------------------------------------------------
# Residual-stream activation patching (H3, ROME/IOI-style)
# ---------------------------------------------------------------------------


def _make_residual_patch_hook(cached: torch.Tensor):
    def hook(_module, _args, output):
        # HF decoder layers return either a Tensor or a tuple whose first
        # element is the hidden state. Replace just that slot.
        if isinstance(output, tuple):
            return (cached,) + output[1:]
        return cached

    return hook


@contextmanager
def residual_patch(
    layers: Sequence[nn.Module],
    layer_idx: int,
    cached_output: torch.Tensor,
) -> Iterator[None]:
    """Replace layer ``layer_idx``'s output hidden state with ``cached_output``.

    The cached tensor must match the hidden-state shape that the patched
    forward will produce (typically ``[batch, seq, d_model]``).
    """
    handle = layers[layer_idx].register_forward_hook(
        _make_residual_patch_hook(cached_output)
    )
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Residual capture (clean-run cache)
# ---------------------------------------------------------------------------


class ResidualCache:
    """Capture per-layer decoder outputs during a forward pass.

    Usage::

        cache = ResidualCache(lm.layers)
        with cache:
            lm.model(input_ids=ids)
        clean = cache.outputs   # {layer_idx: Tensor[batch, seq, d_model]}
    """

    def __init__(self, layers: Sequence[nn.Module], layer_indices: Optional[Sequence[int]] = None):
        self.layers = layers
        self.layer_indices = (
            list(layer_indices) if layer_indices is not None else list(range(len(layers)))
        )
        self.outputs: Dict[int, torch.Tensor] = {}
        self._handles: List = []

    def _make_hook(self, idx: int):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            # Detach + clone so subsequent passes don't reuse the same storage.
            self.outputs[idx] = tensor.detach().clone()

        return hook

    def __enter__(self):
        self.outputs = {}
        for idx in self.layer_indices:
            self._handles.append(self.layers[idx].register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False
