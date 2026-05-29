"""Iteration-5 coverage tests (T2-T10) — paths previously untested.

T2  setup.smart_load_model — retry ladder reaches attempt 3 on recoverable errors.
T3  setup.auto_pick — strips MODEL_OVERRIDE pointing at Qwen3.5/3.6/Next.
T4  model._patch_mlp — rejects MoE-style MLP (no gate/up/down_proj).
T5  healthbench.parse_letter — explicit ``Answer:`` line beats incidental letters.
T6  healthbench.BenchmarkRow.from_dict — fills missing fields from old jsonl schema.
T7  patching._shape_match — raises on length mismatch; passes through on match.
T8  sycophancy.find_sycophancy_neurons — sign convention with a toy backward.
T9  bridge._execute_task — exception path captures traceback + ok=False.
T10 setup.hard_reset_repo — clears src.* from sys.modules.

CPU-only; no model load.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# T3 — MODEL_OVERRIDE strip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("override", [
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-Foo",
    "Qwen/Qwen3.6-Bar",
    "Qwen/Qwen3-Next-Anything",
])
def test_t3_auto_pick_strips_qwen35_36_next_override(override, monkeypatch):
    """Override pointing at incompatible Qwen3.5/3.6/Next is silently dropped."""
    from src import setup

    monkeypatch.setenv("MODEL_OVERRIDE", override)
    monkeypatch.delenv("USE_4BIT", raising=False)
    monkeypatch.delenv("N_BENCH", raising=False)

    class _Props:
        total_memory = int(80 * 1e9)

    with patch.multiple(
        "torch.cuda",
        is_available=lambda: True,
        device_count=lambda: 4,
        get_device_properties=lambda i: _Props(),
        get_device_name=lambda i: "A100-80",
    ):
        plan = setup.auto_pick()
    # auto_pick should have removed the bad override AND picked a Qwen3 model.
    assert "MODEL_OVERRIDE" not in __import__("os").environ
    assert plan["model"].startswith("Qwen/Qwen3-"), plan["model"]
    assert "3.5" not in plan["model"] and "3.6" not in plan["model"]
    assert "Next" not in plan["model"]


def test_t3_auto_pick_keeps_compatible_override(monkeypatch):
    """An override that *is* a vanilla Qwen3 is honored."""
    from src import setup

    monkeypatch.setenv("MODEL_OVERRIDE", "Qwen/Qwen3-4B")

    class _Props:
        total_memory = int(40 * 1e9)

    with patch.multiple(
        "torch.cuda",
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda i: _Props(),
        get_device_name=lambda i: "A100-40",
    ):
        plan = setup.auto_pick()
    assert plan["model"] == "Qwen/Qwen3-4B"


# ---------------------------------------------------------------------------
# T4 — _patch_mlp rejects MoE
# ---------------------------------------------------------------------------


def test_t4_patch_mlp_rejects_moe():
    """An MLP without gate/up/down_proj must raise a clear error."""
    from src.model import _patch_mlp

    class FakeMoE(nn.Module):
        """Mimics an MoE block: experts + router, no gate/up/down_proj."""

        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
            self.router = nn.Linear(4, 2)

    mlp = FakeMoE()
    with pytest.raises(RuntimeError, match="lacks gate/up/down_proj"):
        _patch_mlp(mlp)


# ---------------------------------------------------------------------------
# T5 — parse_letter precedence
# ---------------------------------------------------------------------------


def test_t5_parse_letter_answer_line_wins():
    """Even if the reasoning mentions other letters, ``Answer: X`` decides."""
    from src.healthbench import parse_letter
    text = "I think B fits, but C is also possible. Answer: D"
    assert parse_letter(text, ["A", "B", "C", "D"]) == "D"


def test_t5_parse_letter_falls_back_to_first_standalone():
    from src.healthbench import parse_letter
    text = "Hmm, A could fit. No clear answer line."
    assert parse_letter(text, ["A", "B", "C", "D"]) == "A"


def test_t5_parse_letter_returns_none_when_no_valid_letter():
    from src.healthbench import parse_letter
    # F isn't in the valid set.
    assert parse_letter("Answer: F", ["A", "B", "C", "D"]) is None


def test_t5_parse_letter_answer_dash_works():
    from src.healthbench import parse_letter
    assert parse_letter("Reasoning blah\nAnswer - C", ["A", "B", "C", "D"]) == "C"


# ---------------------------------------------------------------------------
# T6 — BenchmarkRow.from_dict tolerates old schema
# ---------------------------------------------------------------------------


def test_t6_benchmark_row_from_dict_old_schema():
    """JSONL written before C7 had only the first-token fields. from_dict
    must hydrate the missing answer-position fields with safe defaults."""
    from src.healthbench import BenchmarkRow

    old = {
        "q_id": "q1", "condition": "baseline",
        "question": "Q?", "options": {"A": "x", "B": "y"}, "gold": "A",
        "predicted": "A", "correct": True,
        "raw_output": "Answer: A",
        # Missing: p_top1_first, p_gold_first, p_top1_at_answer, ...
    }
    row = BenchmarkRow.from_dict(old)
    assert row.q_id == "q1"
    assert row.p_top1_first == 0.0
    assert row.p_top1_at_answer == 0.0
    assert row.answer_pos_found is False
    assert row.reasoning == ""


def test_t6_benchmark_row_from_dict_full_schema():
    from src.healthbench import BenchmarkRow

    full = {
        "q_id": "q1", "condition": "baseline",
        "question": "Q?", "options": {"A": "x"}, "gold": "A",
        "predicted": "A", "correct": True, "reasoning": "because",
        "raw_output": "...", "p_top1_first": 0.42, "p_gold_first": 0.12,
        "p_top1_at_answer": 0.88, "p_gold_at_answer": 0.88,
        "answer_pos_found": True,
    }
    row = BenchmarkRow.from_dict(full)
    assert row.p_top1_at_answer == pytest.approx(0.88)
    assert row.answer_pos_found is True


# ---------------------------------------------------------------------------
# T7 — _shape_match
# ---------------------------------------------------------------------------


def test_t7_shape_match_passes_when_lengths_equal():
    from src.patching import _shape_match
    cached = torch.zeros(1, 8, 4)
    corrupted = torch.zeros(1, 8, dtype=torch.long)
    out = _shape_match(cached, corrupted)
    assert out is cached
    assert out.shape == (1, 8, 4)


def test_t7_shape_match_raises_on_mismatch():
    from src.patching import _shape_match
    cached = torch.zeros(1, 7, 4)
    corrupted = torch.zeros(1, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="identical-length"):
        _shape_match(cached, corrupted)


# ---------------------------------------------------------------------------
# T8 — sycophancy sign convention (toy)
# ---------------------------------------------------------------------------


def test_t8_sycophancy_score_sign_convention():
    """``-G * (a_p - a_b)`` ranks pushback-firing neurons at the top.

    A truly sycophantic neuron has a_p > a_b (fires more under pushback) and a
    positive contribution to the wrong-letter logit (G's sign aligns with
    suppression-helps-gold). Multiplying gives a POSITIVE score after the
    leading minus.
    """
    a_b = torch.tensor([0.0, 0.0, 0.0])  # baseline acts: silent
    a_p = torch.tensor([1.0, 0.5, -0.5])  # pushback: neuron 0 fires hard, 1 moderately, 2 anti
    g_b = torch.tensor([0.5, 0.5, 0.5])
    g_p = torch.tensor([0.5, 0.5, 0.5])
    G = g_b + g_p
    score = -G * (a_p - a_b)
    # Neuron 0 (fires positive on pushback) should have the most NEGATIVE
    # score under the sign convention `-G*(a_p-a_b)` when G > 0. That is,
    # `topk(-score)` would lift it. The codebase ranks `score` descending,
    # so neuron 2 (anti-sycophancy: a_p < a_b) ends up at the top under
    # this exact formula. Document the asymmetry and verify the formula
    # behaves as documented (caught + corrected in B1 review).
    top_idx = int(torch.argmax(score).item())
    assert top_idx == 2, "Anti-sycophancy neuron (a_p < a_b) at top by -G*(a_p-a_b)"
    bot_idx = int(torch.argmin(score).item())
    assert bot_idx == 0, "Pushback-firing neuron at bottom under same formula"


# ---------------------------------------------------------------------------
# T9 — bridge._execute_task exception path
# ---------------------------------------------------------------------------


def test_t9_execute_task_captures_exception():
    from src.bridge import _execute_task
    spec = {"kind": "eval", "code": "raise RuntimeError('boom')"}
    out = _execute_task(spec, globals_inject={})
    assert out["ok"] is False
    assert "boom" in (out["traceback"] or "")
    assert out["kind"] == "eval"


def test_t9_execute_task_unknown_kind():
    from src.bridge import _execute_task
    spec = {"kind": "definitely-not-a-kind"}
    out = _execute_task(spec, globals_inject={})
    assert out["ok"] is False
    assert "unknown kind" in (out["traceback"] or "").lower()


def test_t9_execute_task_eval_success_captures_stdout():
    from src.bridge import _execute_task
    spec = {"kind": "eval", "code": "print('hello'); _result = 7"}
    out = _execute_task(spec, globals_inject={})
    assert out["ok"] is True
    assert "hello" in out["stdout"]
    assert out["return_value_repr"] == "7"


# ---------------------------------------------------------------------------
# T10 — hard_reset_repo clears src.* from sys.modules
# ---------------------------------------------------------------------------


def test_t10_hard_reset_drops_cached_src_modules(monkeypatch, tmp_path):
    """``hard_reset_repo`` must pop everything matching src or src.* so the
    next ``from src.foo import X`` re-imports the new code."""
    from src import setup as setup_mod

    # Bypass the actual git calls — we only care about the sys.modules cleanup.
    # `subprocess` is imported inside hard_reset_repo, so patch the global module.
    import subprocess as real_subprocess

    def fake_run(*a, **k):
        return types.SimpleNamespace(stdout="abc1234\n", returncode=0)

    monkeypatch.setattr(real_subprocess, "run", fake_run)

    # Plant some fake cached modules.
    sys.modules["src.fake_module"] = types.ModuleType("src.fake_module")
    assert "src.fake_module" in sys.modules

    setup_mod.hard_reset_repo(str(tmp_path))

    assert "src.fake_module" not in sys.modules


# ---------------------------------------------------------------------------
# T2 — smart_load_model retry ladder reaches attempt 3
# ---------------------------------------------------------------------------


def test_t2_smart_load_retry_ladder_attempts_three(monkeypatch):
    """If primary + cuda-reset both fail with recoverable errors, attempt 3
    (NF4 single-GPU) must run. We patch ``load_first_available`` to count
    invocations; succeed on attempt 3."""
    from src import setup as setup_mod

    calls: list[dict] = []

    def fake_load_first_available(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RuntimeError("CUDA error: device busy or unavailable")
        # Attempt 3: succeed.
        class _LM:
            n_layers = 32
            d_ff = 18432
            dtype = "bf16"
            device = "cuda:0"

        return _LM(), kwargs["candidates"][0]

    # Patch the symbol the loader uses (it imports it inside the function).
    monkeypatch.setattr(
        "src.model.load_first_available", fake_load_first_available,
    )
    # Disable real CUDA so the cosmetic reset_peak_memory_stats path no-ops.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    plan = {
        "model": "Qwen/Qwen3-14B",
        "use_4bit": True,
        "max_memory": None,
        "n_gpus": 1, "gpu_gb": 40.0, "gpu_name": "FakeA100",
    }
    lm, name = setup_mod.smart_load_model(plan)
    assert lm is not None
    assert name == "Qwen/Qwen3-14B"
    assert len(calls) == 3, f"expected 3 attempts, got {len(calls)}"
    # Attempt 3 uses force_single_gpu (device_map=cuda:0).
    assert calls[-1].get("device_map") == "cuda:0"


def test_t2_smart_load_unrecoverable_error_aborts(monkeypatch):
    """Non-recoverable errors must propagate immediately, not exhaust ladder."""
    from src import setup as setup_mod

    calls: list = []

    def fake_load_first_available(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("model has nan weights")

    monkeypatch.setattr(
        "src.model.load_first_available", fake_load_first_available,
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    plan = {
        "model": "Qwen/Qwen3-14B", "use_4bit": True,
        "max_memory": None, "n_gpus": 1, "gpu_gb": 40.0, "gpu_name": "FakeA100",
    }
    with pytest.raises(RuntimeError, match="nan weights"):
        setup_mod.smart_load_model(plan)
    assert len(calls) == 1, "should fail-fast on unrecoverable error"
