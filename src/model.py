"""Model loading + MLP forward patching.

Exposes the pre-down-projection activation
    h = SiLU(W_gate(x)) * W_up(x)   (paper, before §2.3)
on every decoder layer's MLP, with ``h.retain_grad()`` so the gradient signal
needed by Eq. 3 is recoverable. Works for any HF causal LM whose MLP module
has ``gate_proj``, ``up_proj``, ``down_proj`` (Llama, Llama-3, Qwen2/3, Mistral).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import SEED


def set_seed(seed: int = SEED) -> None:
    """Deterministic seeding for torch / numpy / random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Patched MLP forward
# ---------------------------------------------------------------------------


def _patched_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """SwiGLU forward that stashes ``h`` (pre-``down_proj``) on the module.

    h = SiLU(W_gate x) * (W_up x)
    Stashed at ``self._h`` with ``retain_grad`` so contrastive gradient
    signals (Eq. 3) can be read after a single ``loss.backward()``.
    """
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    h = F.silu(gate) * up
    if torch.is_grad_enabled() and h.requires_grad:
        h.retain_grad()
    # Always stash for the activation reader (works in inference mode too).
    self._h = h
    return self.down_proj(h)


@dataclass
class LoadedModel:
    """Container returned by :func:`load_model`.

    Attributes
    ----------
    model: HF causal LM, MLPs monkey-patched (``forward``) to expose ``_h``.
    tokenizer: matching tokenizer (left-padded for batched generation).
    layers: list of decoder layers — ``layers[L].mlp._h`` is the per-batch
        activation after the most recent forward through layer L.
    n_layers: number of transformer blocks (``len(layers)``).
    d_ff: MLP intermediate dimension (``W_up.out_features``).
    device, dtype: where the model lives.
    """

    model: nn.Module
    tokenizer: Any
    layers: List[nn.Module]
    n_layers: int
    d_ff: int
    device: torch.device
    dtype: torch.dtype


def _resolve_layers(model: nn.Module) -> List[nn.Module]:
    """Locate the decoder-layer list across HF model layouts."""
    base = getattr(model, "model", model)
    for attr in ("layers", "h", "decoder"):
        layers = getattr(base, attr, None)
        if isinstance(layers, (nn.ModuleList, list)):
            return list(layers)
    raise RuntimeError("Could not locate decoder layer list on this model.")


def _patch_mlp(mlp: nn.Module) -> None:
    """Bind :func:`_patched_mlp_forward` to ``mlp.forward``."""
    if not all(hasattr(mlp, a) for a in ("gate_proj", "up_proj", "down_proj")):
        raise RuntimeError(
            f"MLP {type(mlp).__name__} lacks gate/up/down_proj — "
            "unsupported architecture for this method."
        )
    import types

    mlp.forward = types.MethodType(_patched_mlp_forward, mlp)


def load_model(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = True,
    token: str | None = None,
) -> LoadedModel:
    """Load a causal LM and patch every MLP to expose ``h`` with ``retain_grad``.

    bf16 throughout (do not silently fall back to fp16/fp32 — see CLAUDE.md).
    """
    set_seed()
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, token=token
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    model.eval()

    layers = _resolve_layers(model)
    for layer in layers:
        _patch_mlp(layer.mlp)

    # Probe d_ff from W_up of layer 0.
    d_ff = layers[0].mlp.up_proj.out_features
    device = next(model.parameters()).device
    return LoadedModel(
        model=model,
        tokenizer=tok,
        layers=layers,
        n_layers=len(layers),
        d_ff=d_ff,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def get_h(mlp: nn.Module) -> torch.Tensor:
    """Return the most recently stashed ``h`` tensor for an MLP module."""
    h = getattr(mlp, "_h", None)
    if h is None:
        raise RuntimeError("No cached h — run a forward pass first.")
    return h


def clear_h(layers: Iterable[nn.Module]) -> None:
    """Drop cached ``h`` references to free graph memory between batches."""
    for layer in layers:
        if hasattr(layer.mlp, "_h"):
            layer.mlp._h = None


def forward_with_h(
    lm: LoadedModel, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> Any:
    """Forward pass with ``output_hidden_states=False`` but ``h`` cached on each MLP.

    Caller is responsible for retrieving ``layer.mlp._h`` and calling
    ``.backward()`` if gradients are needed.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return lm.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
