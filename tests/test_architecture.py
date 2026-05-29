"""Iteration-4 architecture regression tests (A1, A2, A4, A5).

A1 — actcache.ActCache round-trip + manifest persistence.
A2 — scripts/push_task.py dry-run writes a well-formed task JSON.
A4 — _letter_token_id_sets memoizes per tokenizer instance.
A5 — find_answer_token_pos is the canonical name + alias of the private fn.

All run CPU-only with no model load.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# A1 — ActCache
# ---------------------------------------------------------------------------


def test_a1_actcache_set_get_round_trip(tmp_path):
    from src.actcache import ActCache, CacheKey
    cache = ActCache(tmp_path)
    k = CacheKey(q_id="q1", condition="baseline", where="ans_pos")
    layers = {3: torch.randn(8), 4: torch.randn(8)}
    assert not cache.has(k)
    cache.set(k, layers)
    assert cache.has(k)
    got = cache.get(k)
    assert got is not None
    assert set(got.keys()) == {3, 4}
    assert got[3].shape == (8,)


def test_a1_actcache_manifest_persists(tmp_path):
    from src.actcache import ActCache, CacheKey
    cache1 = ActCache(tmp_path)
    k = CacheKey("q42", "ablate_overconf", "ans_pos")
    cache1.set(k, {5: torch.randn(4)})
    del cache1
    # New cache instance over same root reads the manifest.
    cache2 = ActCache(tmp_path)
    assert cache2.has(k)
    keys = list(cache2.keys())
    assert any(ck.q_id == "q42" for ck in keys)


def test_a1_actcache_stats(tmp_path):
    from src.actcache import ActCache, CacheKey
    cache = ActCache(tmp_path)
    for i in range(3):
        cache.set(
            CacheKey(f"q{i}", "baseline", "ans_pos"),
            {0: torch.randn(2)},
        )
    s = cache.stats()
    assert s["entries"] == 3


# ---------------------------------------------------------------------------
# A2 — scripts/push_task.py dry-run
# ---------------------------------------------------------------------------


def test_a2_push_task_eval_dry_run(tmp_path, monkeypatch):
    """``push_task eval --dry-run`` writes the queue JSON, doesn't push."""
    # Redirect QUEUE to a tmp dir by monkey-patching the script's ROOT
    # via env-var trick: easier to invoke the parser in-process.
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    import push_task as pt
    importlib.reload(pt)

    monkeypatch.setattr(pt, "QUEUE", tmp_path)
    pt.main([
        "--dry-run", "--label", "smoke",
        "eval", "--code", "1 + 1",
        "--needs", "lm",
        "--save", "out=results/x.json",
    ])
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    spec = json.loads(files[0].read_text())
    assert spec["kind"] == "eval"
    assert spec["code"] == "1 + 1"
    assert spec["needs"] == ["lm"]
    assert spec["save"] == {"out": "results/x.json"}
    assert spec["task_id"].endswith("-smoke")


def test_a2_push_task_call_dry_run(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    import push_task as pt
    importlib.reload(pt)

    monkeypatch.setattr(pt, "QUEUE", tmp_path)
    pt.main([
        "--dry-run",
        "call", "--module", "src.sycophancy",
        "--function", "run_sycophancy_probe",
        "--args", '{"n_questions": 5}',
        "--needs", "lm,items",
    ])
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    spec = json.loads(files[0].read_text())
    assert spec["kind"] == "call"
    assert spec["module"] == "src.sycophancy"
    assert spec["function"] == "run_sycophancy_probe"
    assert spec["args"] == {"n_questions": 5}
    assert spec["needs"] == ["lm", "items"]


# ---------------------------------------------------------------------------
# A4 — letter-token set memoization
# ---------------------------------------------------------------------------


class _ToyTokenizer:
    """Counts forward calls so the test can assert memoization."""

    def __init__(self):
        self.calls = 0

    def __call__(self, s, add_special_tokens=False):
        self.calls += 1
        class _R:
            def __init__(self, ids):
                self.input_ids = ids
        # Stable per-letter ids: ord(L) + len(prefix).
        L = s.strip()[:1]
        if not L:
            return _R([])
        return _R([ord(L) * 10 + len(s) - len(s.lstrip())])


def test_a4_letter_token_id_sets_memoizes():
    from src.healthbench import _letter_token_id_sets
    tok = _ToyTokenizer()
    letters = ["A", "B", "C", "D"]
    out1 = _letter_token_id_sets(tok, letters)
    calls_after_first = tok.calls
    out2 = _letter_token_id_sets(tok, letters)
    # Second call must hit the cache: no new tokenizer calls.
    assert tok.calls == calls_after_first
    assert out1 is out2  # cached dict reused


def test_a4_letter_token_id_sets_distinct_letters_distinct_entries():
    from src.healthbench import _letter_token_id_sets
    tok = _ToyTokenizer()
    out_4 = _letter_token_id_sets(tok, ["A", "B", "C", "D"])
    calls_4 = tok.calls
    out_5 = _letter_token_id_sets(tok, ["A", "B", "C", "D", "E"])
    # Different key → cache miss → tokenizer was called again.
    assert tok.calls > calls_4
    assert set(out_4.keys()) == {"A", "B", "C", "D"}
    assert set(out_5.keys()) == {"A", "B", "C", "D", "E"}


# ---------------------------------------------------------------------------
# A5 — find_answer_token_pos canonical alias
# ---------------------------------------------------------------------------


def test_a5_public_alias_is_same_function():
    from src.healthbench import _find_answer_token_pos, find_answer_token_pos
    assert find_answer_token_pos is _find_answer_token_pos
