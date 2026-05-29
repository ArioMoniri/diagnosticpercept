"""Iteration-6 regression tests for the second-round reviewer findings.

Covers:
  - ActCache atomic write + concurrent set safety.
  - M4 prompt_template threading through run_conditions and parallel worker.
  - H5-at-scale passes length_binned default.
  - H7 length-binned uses Fisher-z (no n_min single-r t-test).
  - Sycophancy summarize_probe reports agreement_corrects_when_wrong.
  - Consensus M6 regex catches post-form commitment phrasings.
  - find_answer_token_pos public alias is the import path everywhere.
  - push_task.py replay strips with an allowlist (no log pollution).
"""
from __future__ import annotations

import inspect
import json
import re
import sys
import threading
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# ActCache atomicity + concurrency
# ---------------------------------------------------------------------------


def test_actcache_set_uses_atomic_write(tmp_path):
    """The on-disk file should never be a partial torch.save dump."""
    from src.actcache import ActCache, CacheKey
    cache = ActCache(tmp_path)
    k = CacheKey("q-a", "baseline", "ans_pos")
    cache.set(k, {0: torch.zeros(4)})
    # Round-trip must succeed (and no .tmp leftovers).
    got = cache.get(k)
    assert got is not None
    leftover = list((tmp_path / "act").glob("*.tmp"))
    assert leftover == [], f"orphan tmp files: {leftover}"


def test_actcache_concurrent_set_no_orphan_tmp(tmp_path):
    """Two threads racing the same key leave a single .pt and no .tmp."""
    from src.actcache import ActCache, CacheKey

    cache = ActCache(tmp_path)
    k = CacheKey("q-race", "baseline", "ans_pos")

    def writer(seed):
        torch.manual_seed(seed)
        cache.set(k, {0: torch.randn(8)})

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    pt_files = list((tmp_path / "act").glob("*.pt"))
    tmp_files = list((tmp_path / "act").glob("*.tmp"))
    assert len(pt_files) == 1, pt_files
    assert tmp_files == [], tmp_files
    got = cache.get(k)
    assert got is not None and 0 in got


# ---------------------------------------------------------------------------
# M4 prompt_template wiring
# ---------------------------------------------------------------------------


def test_run_conditions_accepts_prompt_template_kw():
    """The public ``run_conditions`` and ``run_one`` must take ``prompt_template``."""
    from src.healthbench import run_conditions, run_one
    rc_sig = inspect.signature(run_conditions)
    ro_sig = inspect.signature(run_one)
    assert "prompt_template" in rc_sig.parameters
    assert "prompt_template" in ro_sig.parameters


def test_run_conditions_parallel_accepts_prompt_template_kw():
    from src.parallel import run_conditions_parallel, _worker_main
    p_sig = inspect.signature(run_conditions_parallel)
    w_sig = inspect.signature(_worker_main)
    assert "prompt_template" in p_sig.parameters
    assert "prompt_template" in w_sig.parameters


# ---------------------------------------------------------------------------
# H5-at-scale forwards length_binned
# ---------------------------------------------------------------------------


def test_find_overconfidence_at_scale_defaults_to_length_binned():
    from src.calibration import find_overconfidence_neurons_at_scale
    sig = inspect.signature(find_overconfidence_neurons_at_scale)
    assert "length_binned" in sig.parameters
    assert sig.parameters["length_binned"].default is True


# ---------------------------------------------------------------------------
# H7 Fisher-z aggregation
# ---------------------------------------------------------------------------


def test_h7_length_binned_uses_fisher_z(capsys):
    """The length-binned ranker logs the Fisher-z + bin-sizes line."""
    import random

    from src.h7_layers import rank_miscalibration_neurons

    random.seed(0)
    rows = [
        {
            "q_id": f"q{i}", "gold": "A",
            "predicted": "A" if i % 2 else "B",
            "p_top1_at_answer": random.random(),
            "p_gold_at_answer": random.random(),
            "correct": (i % 2 == 0),
            "chain_len": random.randint(5, 200),
        }
        for i in range(60)
    ]
    acts = {3: torch.randn(60, 8)}
    _ = rank_miscalibration_neurons(rows, acts, top_k=5, fdr_q=0.5,
                                    length_binned=True)
    out = capsys.readouterr().out
    assert "Fisher-z" in out and "bin sizes" in out


# ---------------------------------------------------------------------------
# Sycophancy summarize_probe agreement_corrects_when_wrong
# ---------------------------------------------------------------------------


def test_summarize_probe_reports_agreement_corrects_when_wrong():
    """The new orthogonal metric is in the summary dict and the n_baseline_wrong
    count is reported alongside."""
    from src.sycophancy import SycophancyCase, summarize_probe

    def _case(q_id, base_pred, gold, agr_pred, base_corr):
        return SycophancyCase(
            q_id=q_id, gold=gold, baseline_pred=base_pred,
            baseline_p_top1=0.8, baseline_correct=base_corr,
            wrong_letter="C",
            authority_pred=base_pred, authority_p_top1=0.7,
            authority_flipped_to_user=False,
            insistence_pred=base_pred, insistence_p_top1=0.6,
            insistence_flipped_to_user=False,
            agreement_pred=agr_pred, agreement_p_top1=0.8,
            agreement_flipped_from_baseline=(base_pred != agr_pred),
            baseline_class="committed_letter",
            authority_class="committed_letter",
            insistence_class="committed_letter",
            agreement_class="committed_letter",
            baseline_text="", authority_text="",
            insistence_text="", agreement_text="",
        )

    cases = [
        _case("q1", "B", "A", "A", False),   # wrong → agree corrects to A: counts
        _case("q2", "B", "A", "B", False),   # wrong → still B: doesn't correct
        _case("q3", "A", "A", "A", True),    # already correct: excluded
    ]
    s = summarize_probe(cases)
    assert "agreement_corrects_when_wrong" in s
    assert s["n_baseline_wrong"] == 2
    assert s["agreement_corrects_when_wrong"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Consensus M6 post-form regex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("After review, D is the correct answer.",                     "D"),
    ("B is the best choice given the findings.",                   "B"),
    ("C is my diagnosis.",                                          "C"),
    ("Looking carefully, A is the most likely answer.",            "A"),
    ("My pick is D for this one.",                                  "D"),
    ("answer = C",                                                  "C"),
])
def test_m6_post_form_commitment_regex(text, expected):
    from src.consensus import implied_letter
    assert implied_letter(text, ["A", "B", "C", "D"]) == expected


def test_m6_combines_prefix_and_post_picks_last_by_position():
    """Mixed forms: the latest commitment wins."""
    from src.consensus import implied_letter
    text = "I think A is the best choice. On reflection, the answer is C."
    # "C" appears later in the string; should win.
    assert implied_letter(text, ["A", "B", "C", "D"]) == "C"


# ---------------------------------------------------------------------------
# find_answer_token_pos adoption
# ---------------------------------------------------------------------------


def test_no_private_find_answer_token_pos_imports():
    """H7/H8/sycophancy must import the public alias, not the underscore name."""
    src_root = ROOT / "src"
    offenders = []
    pat = re.compile(r"\bimport\s+_find_answer_token_pos\b|"
                     r"\b_find_answer_token_pos\b")
    for py in src_root.rglob("*.py"):
        if py.name == "healthbench.py":
            continue  # the definition + alias live here.
        text = py.read_text()
        if pat.search(text):
            offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"still using private name: {offenders}"


# ---------------------------------------------------------------------------
# push_task.py replay allowlist
# ---------------------------------------------------------------------------


def test_push_task_replay_strips_log_fields(tmp_path, monkeypatch):
    """A log JSON containing log-only fields must be cleaned before re-enqueue."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    import push_task as pt
    importlib.reload(pt)

    monkeypatch.setattr(pt, "QUEUE", tmp_path)
    log_path = tmp_path / "20260101-000000-foo.json"
    log_path.write_text(json.dumps({
        "task_id": "old-id",
        "kind": "eval", "code": "1+1",
        "needs": ["lm"], "save": {},
        # Log-only fields that must NOT leak into the new spec.
        "ok": True, "stdout": "noise", "stderr": "",
        "return_value_repr": "2", "traceback": None,
        "new_globals": [], "wall_seconds": 0.42,
    }))
    pt.main(["--dry-run", "replay", str(log_path)])
    queued = [p for p in tmp_path.glob("*.json") if p != log_path]
    assert len(queued) == 1
    spec = json.loads(queued[0].read_text())
    forbidden = {"ok", "stdout", "stderr", "return_value_repr", "traceback",
                 "new_globals", "wall_seconds"}
    assert forbidden.isdisjoint(spec.keys()), spec
    assert spec["kind"] == "eval"
    assert spec["code"] == "1+1"
    # task_id is set by _write_task (new timestamp), not copied from the log.
    assert spec["task_id"] != "old-id"


def test_push_task_replay_rejects_log_missing_kind(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    import push_task as pt
    importlib.reload(pt)

    monkeypatch.setattr(pt, "QUEUE", tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"stdout": "stuff"}))
    with pytest.raises(SystemExit, match="no `kind`"):
        pt.main(["--dry-run", "replay", str(bad)])
