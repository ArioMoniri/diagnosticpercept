"""Cheap unit tests for src/setup.py auto-pick decision logic.

These run without a GPU — we patch torch.cuda to simulate different
hardware classes (4× A100 40GB, 4× A100 80GB, 4× H100, 1× T4) and
verify the picked plan (model, use_4bit, max_memory) matches expectations.

This is the CI guard rail for plan changes — if a future fix to
auto_pick() regresses any of these cases, the smoke test fails before
push, not after the user re-imports the notebook on Colab.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.setup import auto_pick


def _mock_torch(n_gpus: int, gpu_gb: float, gpu_name: str = "FakeGPU"):
    """Patch torch.cuda to look like a specific hardware config."""

    class _Props:
        total_memory = int(gpu_gb * 1e9)

    cuda = patch.multiple(
        "torch.cuda",
        is_available=lambda: True,
        device_count=lambda: n_gpus,
        get_device_properties=lambda i: _Props(),
        get_device_name=lambda i: gpu_name,
    )
    return cuda


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("USE_4BIT", "MODEL_OVERRIDE", "N_BENCH"):
        monkeypatch.delenv(k, raising=False)


def test_4x_a100_40gb_picks_nf4_no_max_memory():
    """4× A100 40GB → NF4 27B, no max_memory (model fits one GPU)."""
    with _mock_torch(4, 42.4, "NVIDIA A100-SXM4-40GB"):
        plan = auto_pick()
    assert plan["use_4bit"] is True, "A100 40GB must use NF4"
    assert plan["max_memory"] is None, (
        "NF4 model fits one GPU; spreading would just add overhead"
    )
    assert plan["model"] == "Qwen/Qwen3.6-27B"
    assert plan["n_bench"] == 1273


def test_4x_a100_80gb_picks_bf16_with_spread():
    """4× A100 80GB → bf16 27B spread across all 4."""
    with _mock_torch(4, 79.5, "NVIDIA A100-SXM4-80GB"):
        plan = auto_pick()
    assert plan["use_4bit"] is False
    assert plan["max_memory"] is not None
    assert set(plan["max_memory"].keys()) == {0, 1, 2, 3}
    assert plan["max_memory"][0] == "20GiB", "25% of 80 GB per GPU"


def test_4x_h100_80gb_picks_bf16_with_spread():
    with _mock_torch(4, 79.5, "NVIDIA H100 80GB HBM3"):
        plan = auto_pick()
    assert plan["use_4bit"] is False
    assert plan["max_memory"] is not None


def test_1x_a100_80gb_no_max_memory():
    """Single A100 80GB: bf16 27B fits one GPU, no spread needed."""
    with _mock_torch(1, 79.5):
        plan = auto_pick()
    assert plan["use_4bit"] is False
    assert plan["max_memory"] is None


def test_1x_a100_40gb_picks_nf4():
    """Single A100 40GB: bf16 27B doesn't fit → NF4."""
    with _mock_torch(1, 42.4):
        plan = auto_pick()
    assert plan["use_4bit"] is True
    assert plan["max_memory"] is None


def test_1x_t4_picks_smaller_model():
    """T4 16GB: too small even for 9B; falls to 4B-class."""
    with _mock_torch(1, 14.6):
        plan = auto_pick()
    assert plan["use_4bit"] is True
    assert plan["model"] == "Qwen/Qwen3.5-9B"
    assert plan["n_bench"] == 600


def test_user_override_wins():
    """MODEL_OVERRIDE and USE_4BIT env vars short-circuit the plan."""
    with patch.dict("os.environ", {
        "MODEL_OVERRIDE": "Qwen/Qwen3-8B",
        "USE_4BIT": "0",
        "N_BENCH": "50",
    }), _mock_torch(4, 79.5):
        plan = auto_pick()
    assert plan["model"] == "Qwen/Qwen3-8B"
    assert plan["use_4bit"] is False
    assert plan["n_bench"] == 50
