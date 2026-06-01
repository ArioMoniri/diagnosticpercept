"""Tests for src/checkpoint.py — cross-notebook discovery artifact round-trip."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _disc(**over):
    from src.checkpoint import Discovery
    base = dict(
        model_name="Qwen/Qwen3-32B",
        gate_layer=17, gate_neuron=8680, gate_m_star=-80.0, gate_anchor_d=1.2538,
        critical_layer=0, layer_scores=[1.0, 0.5, 0.2],
        halluc_neurons=[{"layer": 34, "neuron": 2981}, {"layer": 35, "neuron": 11457}],
        overconf_neurons=[{"layer": 18, "neuron": 6522}, {"layer": 18, "neuron": 2009}],
        git_sha="abc1234", n_layers=64, d_ff=18432,
    )
    base.update(over)
    return Discovery(**base)


def test_round_trip(tmp_path):
    from src.checkpoint import load_discovery, save_discovery
    d = _disc()
    save_discovery(tmp_path, d)
    d2 = load_discovery(tmp_path)
    assert d2.model_name == "Qwen/Qwen3-32B"
    assert d2.gate_layer == 17 and d2.gate_neuron == 8680
    assert d2.gate_m_star == pytest.approx(-80.0)
    assert d2.critical_layer == 0
    assert d2.halluc_neurons[0] == {"layer": 34, "neuron": 2981}


def test_combined_neurons_property():
    d = _disc()
    combined = d.combined_neurons
    # overconf first, then halluc.
    assert combined[0] == {"layer": 18, "neuron": 6522}
    assert combined[-1] == {"layer": 35, "neuron": 11457}
    assert len(combined) == 4


def test_gate_dict():
    d = _disc()
    g = d.gate_dict()
    assert g == {"layer": 17, "neuron": 8680, "m_star": -80.0, "anchor_d": 1.2538}


def test_missing_file_raises_actionable(tmp_path):
    from src.checkpoint import load_discovery
    with pytest.raises(FileNotFoundError, match="Run 01_discovery"):
        load_discovery(tmp_path)


def test_missing_required_field_raises(tmp_path):
    from src.checkpoint import DISCOVERY_FILENAME, load_discovery
    # Write a partial file lacking gate_neuron.
    (tmp_path / DISCOVERY_FILENAME).write_text(json.dumps({
        "model_name": "Qwen/Qwen3-32B", "gate_layer": 17,
        # gate_neuron missing
        "gate_m_star": -80.0, "gate_anchor_d": 1.0, "critical_layer": 0,
    }))
    with pytest.raises(ValueError, match="missing required field"):
        load_discovery(tmp_path)


def test_unknown_extra_keys_tolerated(tmp_path):
    """A newer writer adds a field; an older reader must not crash."""
    from src.checkpoint import DISCOVERY_FILENAME, load_discovery
    payload = {
        "model_name": "Qwen/Qwen3-14B",
        "gate_layer": 1, "gate_neuron": 2, "gate_m_star": -5.0,
        "gate_anchor_d": 0.5, "critical_layer": 3,
        "future_field_from_v2": {"foo": "bar"},
    }
    (tmp_path / DISCOVERY_FILENAME).write_text(json.dumps(payload))
    d = load_discovery(tmp_path)
    assert d.model_name == "Qwen/Qwen3-14B"
    assert d.extra.get("future_field_from_v2") == {"foo": "bar"}


def test_save_atomic_no_tmp_leftover(tmp_path):
    from src.checkpoint import save_discovery
    save_discovery(tmp_path, _disc())
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


def test_discovery_exists(tmp_path):
    from src.checkpoint import discovery_exists, save_discovery
    assert not discovery_exists(tmp_path)
    save_discovery(tmp_path, _disc())
    assert discovery_exists(tmp_path)


def test_empty_neuron_lists_warn(tmp_path, capsys):
    """A checkpoint with empty H4/H5 neuron lists warns (silent no-op ablations)."""
    from src.checkpoint import load_discovery, save_discovery
    save_discovery(tmp_path, _disc(halluc_neurons=[], overconf_neurons=[]))
    load_discovery(tmp_path)
    out = capsys.readouterr().out
    assert "no H4 hallucination neurons" in out
    assert "no H5 overconfidence neurons" in out
