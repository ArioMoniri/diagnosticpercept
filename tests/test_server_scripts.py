"""Regression tests for the self-contained server bundle.

  - scripts/run_all.py and scripts/analyze_all.py byte-compile (catches the
    Python <3.12 f-string trap that would break the analyzer on an Ubuntu box).
  - scripts/run_on_server.sh passes `bash -n` and carries the safety rails
    (CUDA gate, EXIT-trap cleanup with a WORK value-guard).
  - analyze_all.py runs on a synthetic COMPLETE results dir and on a PARTIAL
    one (no baseline / no comparison.csv) without crashing, and both emit
    strict-valid JSON (no bare NaN/Infinity).
"""
from __future__ import annotations

import csv
import json
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def test_python_scripts_compile_on_pre_312_syntax():
    # py_compile uses the running interpreter's parser; the key guarantee is no
    # same-quote-nested f-strings (a 3.12-only feature). We assert that
    # textually too, since CI may run 3.12+.
    import re
    for fn in ("run_all.py", "analyze_all.py"):
        p = SCRIPTS / fn
        py_compile.compile(str(p), doraise=True)
        for i, line in enumerate(p.read_text().splitlines(), 1):
            assert not re.search(r"f'[^']*\{[^}]*'[^}']*'", line), \
                f"{fn}:{i} has a same-quote nested f-string (breaks Python <3.12)"


def test_server_script_bash_syntax_and_safety_rails():
    sh = SCRIPTS / "run_on_server.sh"
    r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = sh.read_text()
    # CUDA must be enforced (no silent CPU torch) and be MIG-aware.
    assert "torch.cuda.is_available()" in text
    assert "torch reports CUDA unavailable" in text
    assert "CUDA_VISIBLE_DEVICES" in text and "MIG-" in text
    # Everything is logged to a persistent file (survives a tmux/SSH close).
    assert "dp_bootstrap.log" in text and "exec >" in text
    # Cleanup must be a trap (runs on success+failure) with a WORK value-guard.
    assert "trap cleanup EXIT" in text
    assert 'refusing to rm' in text
    # results copied before any delete; copy failure is fatal (keeps WORK).
    assert "results copy to" in text


def _complete_results(tmp: Path):
    h6 = tmp / "h6"; h6.mkdir(parents=True)
    conds = ["baseline", "h5_ablate_overconf", "h3_zero_layer"]
    summary = {}
    for c in conds:
        acc = 0.74 if c != "h3_zero_layer" else 0.28
        summary[c] = {"n": 3, "accuracy": acc, "mean_p_top1_at_answer": 0.99,
                      "mean_p_gold_at_answer": 0.73, "brier_at_answer": 0.25,
                      "answer_position_found_rate": 0.96}
    (h6 / "summary.json").write_text(json.dumps(summary))
    # comparison.csv with the columns the analyzer reads
    cols = ["q_id", "question", "gold"]
    for c in conds:
        cols += [f"{c}_pred", f"{c}_correct", f"{c}_p_top1_first", f"{c}_p_gold_first",
                 f"{c}_p_top1_answer", f"{c}_p_gold_answer", f"{c}_answer_found", f"{c}_reasoning"]
    with (h6 / "comparison.csv").open("w", newline="") as f:
        wr = csv.writer(f); wr.writerow(cols)
        for i in range(3):
            row = [f"q{i}", "Q?", "A"]
            for c in conds:
                row += ["A", "1", "0.99", "0.73", "0.995", "0.73", "1", "reasoning"]
            wr.writerow(row)
    (h6 / "consensus_flip.json").write_text(json.dumps(
        {"report": {"n_total": 3, "n_baseline_wrong": 1, "n_consensus_flips": 0,
                    "fix_rates": {}}}))


def _run_analyzer(results_dir: Path):
    r = subprocess.run([sys.executable, str(SCRIPTS / "analyze_all.py"), str(results_dir)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = results_dir / "analysis" / "report.json"
    assert rep.exists()
    # Strict JSON: reject bare NaN/Infinity.
    json.loads(rep.read_text(),
               parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"bad const {x}")))
    return (results_dir / "analysis" / "report.md").read_text()


def test_analyzer_on_complete_results(tmp_path):
    _complete_results(tmp_path)
    md = _run_analyzer(tmp_path)
    assert "## H6" in md and "Key result" in md   # null-effect paragraph prints


def test_analyzer_on_partial_results_no_baseline_no_csv(tmp_path):
    h6 = tmp_path / "h6"; h6.mkdir(parents=True)
    (h6 / "summary.json").write_text(json.dumps(
        {"h5_ablate_overconf": {"n": 10, "accuracy": 0.7, "mean_p_top1_at_answer": 0.9,
                                "mean_p_gold_at_answer": 0.7, "brier_at_answer": 0.2,
                                "answer_position_found_rate": 0.9}}))
    md = _run_analyzer(tmp_path)        # must not crash; strict JSON checked inside
    assert "H6" in md


def test_analyzer_on_empty_dir(tmp_path):
    md = _run_analyzer(tmp_path)
    assert "Diagnostic Percept" in md
