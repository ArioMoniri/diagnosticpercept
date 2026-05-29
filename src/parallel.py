"""Multi-GPU data parallelism for H6 / H7 / sycophancy.

Single-GPU code paths in ``src.healthbench`` / ``src.sycophancy`` are
embarrassingly parallel — each question is independent and intervention
factories close over per-process ``lm.layers``. So we shard *items* across
workers, each worker loads its own model copy on a dedicated GPU, runs its
chunk through ``run_conditions``, and the main process merges JSONLs.

Why data parallelism (not tensor / pipeline parallelism):

  - Model parallelism would split layers across GPUs. Our patched
    ``LlamaMLP.forward`` stashes ``h`` on each MLP module, but with
    ``device_map='auto'`` those modules live on different GPUs — the
    Python-level hooks still work but the gradient pass and intervention
    composition become tricky to reason about.
  - Data parallelism is one model copy per GPU, ~64 GB per copy at bf16
    for Qwen3.6-27B → fits comfortably in an H100 80 GB. With N=4 H100
    we get a clean 4× wall-time reduction on H6/H7/sycophancy.

Use::

    from src.parallel import run_conditions_parallel, conditions_to_specs

    specs = conditions_to_specs(
        baseline=None,
        h1_gate_anchor=dict(type='anchor', layer=L_star, neuron=N_star,
                             m_star=m_star, d=anchor_d, k=1.0),
        h5_ablate_overconf=dict(type='ablate', neurons=top_overconf),
    )
    run_conditions_parallel(
        model_name=MODEL_NAME, items=items, condition_specs=specs,
        out_dir=H6_RESULTS, n_gpus=torch.cuda.device_count(),
    )
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Condition spec ↔ factory translation
# ---------------------------------------------------------------------------


def conditions_to_specs(**conds) -> Dict[str, Optional[Dict]]:
    """Pass-through helper to declare condition specs.

    Each value should be either ``None`` (baseline) or a dict matching
    the ``type`` schemas understood by :func:`build_conditions_from_specs`.
    """
    return dict(conds)


def build_conditions_from_specs(layers, specs: Dict[str, Optional[Dict]]):
    """Rebuild intervention factories inside a worker process.

    Factories close over the worker's own ``lm.layers``. ``specs`` is the
    JSON-serializable description we passed across the process boundary.
    """
    from .healthbench import (
        ablate_neurons_factory, anchor_factory, zero_mlp_factory,
        additive_shift_factory,
    )
    out: Dict[str, Any] = {}
    for name, spec in specs.items():
        if spec is None:
            out[name] = None
            continue
        kind = spec["type"]
        if kind == "ablate":
            out[name] = ablate_neurons_factory(layers, spec["neurons"])
        elif kind == "zero_mlp":
            out[name] = zero_mlp_factory(layers, spec["layers"])
        elif kind == "anchor":
            out[name] = anchor_factory(
                layers, spec["layer"], spec["neuron"],
                float(spec["m_star"]), float(spec["d"]),
                float(spec.get("k", 1.0)),
            )
        elif kind == "additive_shift":
            out[name] = additive_shift_factory(layers, spec["shifts"])
        else:
            raise ValueError(f"unknown condition type: {kind!r}")
    return out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _worker_main(
    rank: int,
    chunk_items_json: List[Dict],
    condition_specs: Dict[str, Optional[Dict]],
    model_name: str,
    out_dir: str,
    token: Optional[str],
    quantize_4bit: bool,
) -> None:
    """One worker process. Loads model on its assigned GPU, runs items."""
    # Restrict this process to its own GPU. CUDA_VISIBLE_DEVICES must be set
    # before any torch CUDA call in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # Enable expandable segments to avoid fragmentation across many gen calls.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch as _t
    from .healthbench import MCQItem, run_conditions
    from .model import load_model

    _t.cuda.set_device(0)   # index 0 inside the restricted view
    print(f"[worker {rank}] loading {model_name} on GPU{rank}  (4bit={quantize_4bit})")
    lm = load_model(model_name, token=token, quantize_4bit=quantize_4bit)
    if _t.cuda.is_available():
        _t.cuda.empty_cache()
        print(f"[worker {rank}] VRAM after load: "
              f"{_t.cuda.memory_allocated()/1e9:.2f} GB allocated, "
              f"{_t.cuda.memory_reserved()/1e9:.2f} GB reserved")
    conditions = build_conditions_from_specs(lm.layers, condition_specs)

    # Reconstruct items.
    items = [MCQItem(**d) for d in chunk_items_json]
    worker_out = Path(out_dir) / f"_gpu{rank}"
    worker_out.mkdir(parents=True, exist_ok=True)
    print(f"[worker {rank}] running {len(items)} items × {len(conditions)} conditions")

    t0 = time.time()
    run_conditions(lm, items, conditions, out_dir=worker_out)
    print(f"[worker {rank}] done in {(time.time() - t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _split_round_robin(items: Sequence, n: int) -> List[List]:
    chunks: List[List] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return chunks


def _items_to_json(items) -> List[Dict]:
    out = []
    for it in items:
        # MCQItem is a dataclass; asdict handles it. Bare dicts pass through.
        out.append(asdict(it) if hasattr(it, "__dataclass_fields__") else dict(it))
    return out


def run_conditions_parallel(
    model_name: str,
    items: Sequence,
    condition_specs: Dict[str, Optional[Dict]],
    out_dir: Path,
    n_gpus: Optional[int] = None,
    token: Optional[str] = None,
    quantize_4bit: bool = True,
) -> Path:
    """Default ``quantize_4bit=True`` because each worker holds a full model
    copy on its own GPU; even on an 80 GB H100, the bf16 27B-class model
    (~64 GB weights) leaves only ~16 GB for reasoning-chain KV cache and
    activations — too tight, OOM-prone on long generations.

    NF4 weights (~14 GB) leave ~66 GB headroom per worker → robust."""
    """Run condition_specs over ``items`` across ``n_gpus`` data-parallel workers.

    Each worker writes ``<out_dir>/_gpu<rank>/<condition>.jsonl``. After all
    workers finish we merge into ``<out_dir>/<condition>.jsonl``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_gpus is None:
        n_gpus = max(1, torch.cuda.device_count())

    chunks = _split_round_robin(list(items), n_gpus)
    chunks_json = [_items_to_json(c) for c in chunks]
    for i, c in enumerate(chunks):
        print(f"  GPU{i}: {len(c)} items")

    # Spawn processes (spawn ctx for CUDA safety).
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(n_gpus):
        p = ctx.Process(
            target=_worker_main,
            args=(rank, chunks_json[rank], condition_specs, model_name,
                  str(out_dir), token, quantize_4bit),
        )
        p.start()
        procs.append(p)

    failed = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed += 1
            print(f"  worker {p.pid} exited with code {p.exitcode}")
    if failed:
        raise RuntimeError(f"{failed}/{n_gpus} workers failed")

    _merge_worker_outputs(out_dir, condition_specs.keys(), n_gpus)
    return out_dir


def _merge_worker_outputs(out_dir: Path, condition_names, n_gpus: int) -> None:
    """Concatenate per-GPU JSONL into a single file per condition."""
    for cond in condition_names:
        merged = out_dir / f"{cond}.jsonl"
        with merged.open("w") as out:
            for rank in range(n_gpus):
                shard = out_dir / f"_gpu{rank}" / f"{cond}.jsonl"
                if shard.exists():
                    out.write(shard.read_text())
        print(f"  merged → {merged} ({merged.stat().st_size/1024:.1f} KB)")

    # Build the comparison.csv + summary.json using the single-GPU helpers
    # over the merged JSONLs.
    from .healthbench import BenchmarkRow, _write_comparison_csv, _write_summary
    all_results: Dict[str, List[BenchmarkRow]] = {}
    for cond in condition_names:
        rows = []
        merged = out_dir / f"{cond}.jsonl"
        if merged.exists():
            for line in merged.read_text().splitlines():
                if line.strip():
                    rows.append(BenchmarkRow(**json.loads(line)))
        all_results[cond] = rows

    # We need the per-row item-list to rebuild comparison.csv; reuse the
    # q_ids and texts that any row carries.
    q_id_to_item = {}
    for rows in all_results.values():
        for r in rows:
            q_id_to_item.setdefault(r.q_id, type("I", (), {
                "q_id": r.q_id, "question": r.question, "gold": r.gold,
                "options": r.options,
            }))
    items_view = list(q_id_to_item.values())
    _write_comparison_csv(items_view, all_results, out_dir / "comparison.csv")
    _write_summary(all_results, out_dir / "summary.json")
