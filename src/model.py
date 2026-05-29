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
from typing import Any, Iterable, List, Optional, Sequence

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


# Default candidate chain — tried in order, first one that loads wins.
# QWEN-ONLY. If you know the exact newest Qwen repo name, set
# ``MODEL_OVERRIDE='Qwen/<exact-name>'`` to short-circuit the chain.
#
# Naming-convention sweep (since exact May 2026 Qwen3.5 / Qwen3-Next names
# can't be confirmed at training time). Each line tries one plausible
# pattern; non-existent repos just print "not on HF, skipping" and we
# proceed to the next.
DEFAULT_MODEL_CANDIDATES: tuple[str, ...] = (
    # Qwen3 (April 2025) — confirmed loadable on standard transformers.
    # Default chain.
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    # Qwen3.5 (May 2026) — natively multimodal, hybrid Gated DeltaNet /
    # Gated Attention. Architecture not yet in transformers main as of
    # writing; loading currently raises "does not recognize this
    # architecture". Listed here as deeper fallbacks for future builds.
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-4B",
)
QWEN_PREFIX = "Qwen/"


def load_first_available(
    candidates: Sequence[str] = DEFAULT_MODEL_CANDIDATES,
    **kwargs: Any,
) -> tuple["LoadedModel", str]:
    """Try ``candidates`` in order, returning (model, picked_name).

    Each failure prints the exception and proceeds. Raises if every
    candidate fails. Pass ``token=`` if any candidate is a gated model.
    """
    last_err: Exception | None = None
    for name in candidates:
        try:
            print(f"[load_first_available] trying {name} ...")
            lm = load_model(name, **kwargs)
            print(f"[load_first_available] loaded {name}")
            return lm, name
        except Exception as e:
            msg = str(e).split("\n", 1)[0]
            # Quiet "repo not found" — those are expected when probing variants.
            if "is not a local folder and is not a valid model identifier" in msg:
                print(f"[load_first_available]   {name}: not on HF, skipping")
            else:
                print(f"[load_first_available]   {name} failed: {type(e).__name__}: {msg}")
            last_err = e
    raise RuntimeError(
        f"All candidates failed. Last error: {last_err!r}"
    )


def load_model(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = True,
    token: str | None = None,
    quantize_4bit: bool = False,
) -> LoadedModel:
    """Load a causal LM and patch every MLP to expose ``h`` with ``retain_grad``.

    bf16 throughout for the compute path. ``quantize_4bit=True`` stores weights
    in NF4 via bitsandbytes (~5 GB for an 8B model) while keeping the compute
    dtype at bf16 — needed on Colab T4 / L4 (≤16 GB VRAM).

    bitsandbytes ``Linear4bit`` keeps activations in bf16, so ``h.retain_grad``
    still flows correctly through the de-quantization op for H1.
    """
    set_seed()
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, token=token
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = dict(
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        # device_map="auto" is required for 4-bit; remove explicit dtype to
        # avoid HF complaining about dtype on already-quantized weights.
        kwargs.pop("torch_dtype", None)
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
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
