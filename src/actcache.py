"""Shared activation-cache helper for H6 / H7 / H8 / sycophancy.

H6 generates an answer; H7 needs the answer-position activations across
late layers; H8 needs MCQ-position activations *and* a prose-attestation
activation; sycophancy needs activations from the baseline forward. Today
each module runs its own forward pass to capture those activations →
the same Qwen3-32B forward is computed 3-4 times per question.

A1 (ml-developer review 2026-05-29): cache the *answer-position activation
vector per layer* keyed on ``(q_id, condition, "ans_pos")`` so downstream
modules can pull a previously-computed forward instead of re-running it.

Cache is on-disk JSON manifest + per-key ``.pt`` tensors so workers in a
multi-GPU run can share it via the filesystem; in-memory access goes
through ``ActCache.get / set``.

The data parallel workers each see a private VRAM pool but a shared HOME
filesystem in Colab Enterprise, so disk caching is the natural choice
(per-key tensors are 8-12 MB for d_ff=18432 fp16; 1273 questions × 6
conditions × 16 layers ≈ a few GB total, well under the boot disk).
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


def _key_hash(q_id: str, condition: str, where: str) -> str:
    """Filesystem-safe hash of the cache key. Short, collision-resistant."""
    h = hashlib.blake2b(
        f"{q_id}|{condition}|{where}".encode("utf-8"), digest_size=10
    ).hexdigest()
    return h


@dataclass(frozen=True)
class CacheKey:
    """Uniquely identifies one cached activation vector."""

    q_id: str
    condition: str           # "baseline" / "ablate_overconf" / ...
    where: str               # "ans_pos" / "first_tok" / custom tag

    def hashed(self) -> str:
        return _key_hash(self.q_id, self.condition, self.where)


class ActCache:
    """Thread-safe on-disk cache for ``CacheKey → {layer: Tensor}`` mappings.

    Layout::

        <root>/
          manifest.json          # {hashed_key: {q_id, condition, where, layers}}
          act/<hash>.pt          # torch.save({layer: Tensor[d_ff]}, ...)
    """

    def __init__(self, root: Path, in_memory: bool = True):
        self.root = Path(root)
        (self.root / "act").mkdir(parents=True, exist_ok=True)
        self.in_memory = in_memory
        self._lock = threading.Lock()
        self._mem: Dict[str, Dict[int, torch.Tensor]] = {}
        self._manifest_path = self.root / "manifest.json"
        self._manifest: Dict[str, Dict[str, Any]] = (
            json.loads(self._manifest_path.read_text())
            if self._manifest_path.exists() else {}
        )

    # ------------------------------------------------------------------ #
    # Core API                                                           #
    # ------------------------------------------------------------------ #

    def has(self, key: CacheKey) -> bool:
        h = key.hashed()
        with self._lock:
            if self.in_memory and h in self._mem:
                return True
            return h in self._manifest

    def get(self, key: CacheKey) -> Optional[Dict[int, torch.Tensor]]:
        h = key.hashed()
        with self._lock:
            if self.in_memory and h in self._mem:
                return self._mem[h]
        path = self.root / "act" / f"{h}.pt"
        if not path.exists():
            return None
        # torch.load can be invoked outside the lock — single-writer/single-
        # reader semantics per file, and we don't mutate.
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # weights_only kw added in torch 2.4; fall back for older runtimes.
            obj = torch.load(path, map_location="cpu")
        if self.in_memory:
            with self._lock:
                self._mem[h] = obj
        return obj

    def set(self, key: CacheKey, layers: Dict[int, torch.Tensor]) -> None:
        h = key.hashed()
        path = self.root / "act" / f"{h}.pt"
        # Detach + CPU to make the cache portable across worker processes.
        layers_cpu = {int(L): v.detach().to(dtype=torch.float16, device="cpu")
                      for L, v in layers.items()}
        torch.save(layers_cpu, path)
        with self._lock:
            self._manifest[h] = {
                "q_id": key.q_id, "condition": key.condition,
                "where": key.where, "layers": sorted(layers_cpu.keys()),
            }
            self._manifest_path.write_text(json.dumps(self._manifest, indent=2))
            if self.in_memory:
                self._mem[h] = layers_cpu

    def keys(self) -> Iterable[CacheKey]:
        with self._lock:
            for h, meta in self._manifest.items():
                yield CacheKey(meta["q_id"], meta["condition"], meta["where"])

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._manifest),
                "in_memory": len(self._mem) if self.in_memory else 0,
            }


# --------------------------------------------------------------------------- #
# Convenience: shared singleton                                                #
# --------------------------------------------------------------------------- #

_GLOBAL_CACHE: Optional[ActCache] = None


def get_global_cache(root: Optional[Path] = None) -> Optional[ActCache]:
    """Return the process-global ActCache. ``None`` if neither already-set nor
    a ``root`` is provided. Workers can opt-in by calling
    :func:`set_global_cache` once at startup.
    """
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None and root is not None:
        _GLOBAL_CACHE = ActCache(root)
    return _GLOBAL_CACHE


def set_global_cache(cache: Optional[ActCache]) -> None:
    """Install the process-global ActCache (or clear it with ``None``)."""
    global _GLOBAL_CACHE
    _GLOBAL_CACHE = cache
