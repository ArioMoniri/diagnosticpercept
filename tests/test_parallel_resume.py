"""Cross-runtime resume tests: re-sharding + q_id-based resume.

These cover the Colab-Enterprise resume path where a multi-GPU H6 run is
interrupted and resumed in a *different* runtime (possibly with a different
GPU count). No model / torch.cuda needed — run_one is monkeypatched.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# _reseed_shards
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def test_reseed_redistributes_to_new_gpu_count(tmp_path):
    from src.parallel import _reseed_shards

    out = tmp_path / "h6"
    out.mkdir()
    items = [{"q_id": f"q{i}"} for i in range(6)]
    # Prior run was 3-GPU: rank = i % 3 → q0,q3→g0  q1,q4→g1  q2,q5→g2.
    _write_jsonl(out / "_gpu0" / "baseline.jsonl", [{"q_id": "q0"}, {"q_id": "q3"}])
    _write_jsonl(out / "_gpu1" / "baseline.jsonl", [{"q_id": "q1"}, {"q_id": "q4"}])
    _write_jsonl(out / "_gpu2" / "baseline.jsonl", [{"q_id": "q2"}, {"q_id": "q5"}])

    # Resume on 2 GPUs: rank = i % 2 → q0,q2,q4→g0  q1,q3,q5→g1.
    _reseed_shards(out, items, ["baseline"], n_gpus=2)

    g0 = {json.loads(l)["q_id"] for l in (out / "_gpu0" / "baseline.jsonl").read_text().splitlines() if l.strip()}
    g1 = {json.loads(l)["q_id"] for l in (out / "_gpu1" / "baseline.jsonl").read_text().splitlines() if l.strip()}
    assert g0 == {"q0", "q2", "q4"}
    assert g1 == {"q1", "q3", "q5"}
    # No data lost.
    assert g0 | g1 == {f"q{i}" for i in range(6)}
    meta = json.loads((out / "_dp_meta.json").read_text())
    assert meta["n_gpus"] == 2 and meta["n_items"] == 6


def test_reseed_merges_completed_merged_file(tmp_path):
    """Done rows in the merged {cond}.jsonl (from a completed prior run) seed shards."""
    from src.parallel import _reseed_shards

    out = tmp_path / "h6"
    out.mkdir()
    items = [{"q_id": f"q{i}"} for i in range(4)]
    _write_jsonl(out / "baseline.jsonl",
                 [{"q_id": "q0"}, {"q_id": "q1"}, {"q_id": "q2"}, {"q_id": "q3"}])
    _reseed_shards(out, items, ["baseline"], n_gpus=2)
    g0 = {json.loads(l)["q_id"] for l in (out / "_gpu0" / "baseline.jsonl").read_text().splitlines() if l.strip()}
    g1 = {json.loads(l)["q_id"] for l in (out / "_gpu1" / "baseline.jsonl").read_text().splitlines() if l.strip()}
    assert g0 == {"q0", "q2"} and g1 == {"q1", "q3"}


def test_reseed_dedups_and_tolerates_bad_lines(tmp_path):
    from src.parallel import _reseed_shards

    out = tmp_path / "h6"
    out.mkdir()
    items = [{"q_id": f"q{i}"} for i in range(2)]
    # q0 appears in both merged and a shard (dup); a truncated line is present.
    (out / "baseline.jsonl").write_text(
        json.dumps({"q_id": "q0", "v": 1}) + "\n" + '{"q_id": "q1", "v"\n')  # bad 2nd line
    _write_jsonl(out / "_gpu0" / "baseline.jsonl", [{"q_id": "q0", "v": 2}])
    _reseed_shards(out, items, ["baseline"], n_gpus=1)
    rows = [json.loads(l) for l in (out / "_gpu0" / "baseline.jsonl").read_text().splitlines() if l.strip()]
    qids = [r["q_id"] for r in rows]
    assert qids.count("q0") == 1   # de-duplicated


def test_reseed_noop_on_fresh_dir(tmp_path):
    from src.parallel import _reseed_shards
    out = tmp_path / "h6"
    out.mkdir()
    _reseed_shards(out, [{"q_id": "q0"}], ["baseline"], n_gpus=2)
    # No shards created (nothing done), but meta written.
    assert not (out / "_gpu0" / "baseline.jsonl").exists()
    assert (out / "_dp_meta.json").exists()


# ---------------------------------------------------------------------------
# q_id-based resume in run_conditions
# ---------------------------------------------------------------------------


def test_run_conditions_resumes_by_qid_not_position(tmp_path, monkeypatch):
    from src import healthbench as hb

    def make_row(item, cond):
        return hb.BenchmarkRow(
            q_id=item.q_id, condition=cond, question=item.question,
            options=item.options, gold=item.gold, predicted="A", correct=True,
            reasoning="", raw_output="", p_top1_first=0.0, p_gold_first=0.0,
            p_top1_at_answer=0.0, p_gold_at_answer=0.0, answer_pos_found=False,
        )

    items = [hb.MCQItem(q_id=f"q{i}", question="?", options={"A": "a", "B": "b"}, gold="A")
             for i in range(5)]
    out = tmp_path / "h6"
    out.mkdir()
    # Pre-seed q1 and q3 as done, OUT OF ORDER (positional resume would be wrong).
    (out / "baseline.jsonl").write_text(
        json.dumps(dataclasses.asdict(make_row(items[3], "baseline"))) + "\n"
        + json.dumps(dataclasses.asdict(make_row(items[1], "baseline"))) + "\n"
    )

    processed = []

    def fake_run_one(lm, item, cond, intervention_ctx=None, max_new_tokens=512,
                     prompt_template=None):
        processed.append(item.q_id)
        return make_row(item, cond)

    monkeypatch.setattr(hb, "run_one", fake_run_one)
    res = hb.run_conditions(lm=None, items=items, conditions={"baseline": None},
                            out_dir=out)
    # Only the 3 NOT already on disk get processed — regardless of position.
    assert set(processed) == {"q0", "q2", "q4"}
    assert len(res["baseline"]) == 5
    # All five q_ids present exactly once in the final jsonl.
    final = [json.loads(l)["q_id"] for l in (out / "baseline.jsonl").read_text().splitlines() if l.strip()]
    assert sorted(final) == ["q0", "q1", "q2", "q3", "q4"]
