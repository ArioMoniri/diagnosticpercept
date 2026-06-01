"""Cross-notebook discovery checkpoint.

The pipeline is split into separate Colab Enterprise notebooks so each phase
runs in its own runtime session (discovery is cheap; the H6/H7/H8 benchmark
is expensive and must survive a disconnect without re-running H1-H5):

    01_discovery.ipynb   →  writes  results/discovery.json   (this module)
    02_benchmark.ipynb   →  reads   results/discovery.json
    03_scale.ipynb       →  reads   results/discovery.json   (+ h6 outputs)
    04_sycophancy.ipynb  →  reads   results/discovery.json

Every neuron coordinate H6/H7/H8 needs (the H1 gate, H3 critical layer, H4
hallucination neurons, H5 overconfidence neurons, the anchor parameters) is a
small JSON-serializable record. We persist exactly those so a fresh runtime
can rebuild the intervention factories without a GPU pass through discovery.

The JSON is intentionally flat + schema-versioned so a future field addition
doesn't break an in-progress multi-notebook run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1
DISCOVERY_FILENAME = "discovery.json"


@dataclass
class Discovery:
    """Everything the benchmark / scale / sycophancy phases need from H1-H5.

    All fields are plain Python scalars / lists-of-dicts so the record round
    trips through JSON with no custom encoder.
    """

    model_name: str

    # H1 — diagnosis gate.
    gate_layer: int
    gate_neuron: int
    gate_m_star: float
    gate_anchor_d: float

    # H3 — critical routing layer (single layer index) + per-layer scores.
    critical_layer: int
    layer_scores: List[float] = field(default_factory=list)

    # H4 — top hallucination neurons [{layer, neuron}, ...].
    halluc_neurons: List[Dict[str, int]] = field(default_factory=list)

    # H5 — top overconfidence neurons [{layer, neuron}, ...].
    overconf_neurons: List[Dict[str, int]] = field(default_factory=list)

    # Bookkeeping.
    git_sha: str = ""
    created_utc: str = ""          # caller stamps this (Date.now is unavailable here)
    n_layers: int = 0
    d_ff: int = 0
    schema_version: int = SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- derived convenience ----

    @property
    def combined_neurons(self) -> List[Dict[str, int]]:
        """H4 ∪ H5 neuron list used by the `h4_h5_combined` condition."""
        return list(self.overconf_neurons) + list(self.halluc_neurons)

    def gate_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.gate_layer, "neuron": self.gate_neuron,
            "m_star": self.gate_m_star, "anchor_d": self.gate_anchor_d,
        }


_REQUIRED = (
    "model_name", "gate_layer", "gate_neuron", "gate_m_star",
    "gate_anchor_d", "critical_layer",
)


def save_discovery(results_dir: Path | str, disc: Discovery) -> Path:
    """Write ``disc`` to ``<results_dir>/discovery.json`` (pretty JSON).

    Returns the path. Creates ``results_dir`` if needed. Atomic via a
    tempfile + os.replace so a concurrent reader never sees a half-written
    file (a stray Drive sync, for instance).
    """
    import os
    import tempfile

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / DISCOVERY_FILENAME
    payload = json.dumps(asdict(disc), indent=2, sort_keys=False)

    fd, tmp = tempfile.mkstemp(prefix="discovery.", suffix=".json.tmp", dir=str(out_dir))
    os.close(fd)
    try:
        Path(tmp).write_text(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_discovery(results_dir: Path | str) -> Discovery:
    """Read ``<results_dir>/discovery.json`` into a :class:`Discovery`.

    Raises a clear, actionable error if the file is missing (so a benchmark
    notebook tells the user to run discovery first) or if a required field
    is absent (schema mismatch). Unknown extra keys are tolerated — they are
    dropped into ``extra`` so older readers don't crash on newer writers.
    """
    path = Path(results_dir) / DISCOVERY_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No discovery checkpoint at {path}. Run 01_discovery.ipynb first "
            f"(and, on Colab Enterprise, make sure it mirrored results/ to "
            f"Drive/GCS so this runtime can restore it)."
        )
    raw = json.loads(path.read_text())
    missing = [k for k in _REQUIRED if k not in raw or raw[k] is None]
    if missing:
        raise ValueError(
            f"discovery.json at {path} is missing required field(s): {missing}. "
            f"Re-run 01_discovery.ipynb with the current src/ to regenerate it."
        )

    known = set(Discovery.__dataclass_fields__)
    kept = {k: v for k, v in raw.items() if k in known}
    extra = {k: v for k, v in raw.items() if k not in known}
    if extra:
        kept.setdefault("extra", {}).update(extra)
    disc = Discovery(**kept)

    # Soft check: the H4/H5 ablation conditions silently become no-ops if their
    # neuron lists are empty. The save cell always writes them, so an empty list
    # means a truncated/old checkpoint — warn loudly rather than run an empty
    # ablation that looks like "intervention had no effect".
    if not disc.halluc_neurons:
        print(f"!! discovery.json has no H4 hallucination neurons — the "
              f"h4_ablate_halluc / h4_h5_combined conditions will be no-ops. "
              f"Re-run 01_discovery.ipynb.")
    if not disc.overconf_neurons:
        print(f"!! discovery.json has no H5 overconfidence neurons — the "
              f"h5_ablate_overconf / h4_h5_combined conditions will be no-ops. "
              f"Re-run 01_discovery.ipynb.")
    return disc


def discovery_exists(results_dir: Path | str) -> bool:
    return (Path(results_dir) / DISCOVERY_FILENAME).exists()
