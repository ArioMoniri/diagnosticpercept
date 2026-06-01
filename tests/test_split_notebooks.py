"""Regression tests for the split-notebook generator.

These import the generator's builder functions, assemble each phase's cell
list in-memory, and assert:
  * every code cell parses (no syntax errors from the glue),
  * the persistence preamble + mirror are present in every phase,
  * the discovery checkpoint is saved by 01 and restored by 02/03,
  * the committed notebooks/split/*.ipynb files exist and are valid JSON.

Runs CPU-only; imports nothing from torch. Requires the monolith to exist
(scripts/build_notebook.py output), which is committed to the repo.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

nbf = pytest.importorskip("nbformat")


@pytest.fixture(scope="module")
def gen():
    import build_split_notebooks as g
    return g


def _src(cell) -> str:
    return cell.source if isinstance(cell.source, str) else "".join(cell.source)


def _code_sources(cells):
    return [_src(c) for c in cells if c.cell_type == "code"]


def _all_phase_builders(gen):
    return {
        "00": gen.build_00_setup,
        "01": gen.build_01_discovery,
        "02": gen.build_02_benchmark,
        "03": gen.build_03_scale,
        "04": gen.build_04_sycophancy,
    }


def test_every_phase_code_cell_parses(gen):
    for name, build in _all_phase_builders(gen).items():
        for i, src in enumerate(_code_sources(build())):
            try:
                ast.parse(src)
            except SyntaxError as e:  # pragma: no cover - failure path
                pytest.fail(f"phase {name} code cell #{i} syntax error: {e}")


def test_persistence_preamble_in_every_runtime_phase(gen):
    # 01-04 must restore + mirror; 00 is a local health check (no restore).
    for name in ("01", "02", "03", "04"):
        srcs = "\n".join(_code_sources(_all_phase_builders(gen)[name]()))
        assert "persist: restore results/" in srcs, name
        assert "persist: mirror results/" in srcs, name
        assert "detect_backend" in srcs and "build_sync_cmd" in srcs, name


def test_discovery_saved_by_01_restored_by_02_03(gen):
    s01 = "\n".join(_code_sources(gen.build_01_discovery()))
    assert "save_discovery(RESULTS" in s01
    assert "from src.checkpoint import Discovery, save_discovery" in s01

    for name in ("02", "03"):
        srcs = "\n".join(_code_sources(_all_phase_builders(gen)[name]()))
        assert "load_discovery(RESULTS)" in srcs, name


def test_benchmark_frees_main_model_before_parallel(gen):
    srcs = "\n".join(_code_sources(gen.build_02_benchmark()))
    assert "Freed main model before parallel H6" in srcs
    # Must drop BOTH condition dicts (closures pin the weights).
    assert "ALL_CONDITIONS" in srcs and "CONDITIONS" in srcs


def test_no_med_model_references(gen):
    """Qwen-only: no Med42/Med43/meditron/clinical-camel anywhere in the cells."""
    banned = ("med42", "med43", "med-4", "meditron", "clinical-camel")
    for name, build in _all_phase_builders(gen).items():
        blob = "\n".join(_code_sources(build())).lower()
        for b in banned:
            assert b not in blob, f"phase {name} references banned model {b!r}"


def test_committed_split_notebooks_valid_json():
    split_dir = ROOT / "notebooks" / "split"
    expected = {
        "00_setup_check.ipynb", "01_discovery.ipynb", "02_benchmark.ipynb",
        "03_scale.ipynb", "04_sycophancy.ipynb",
    }
    present = {p.name for p in split_dir.glob("*.ipynb")}
    assert expected <= present, f"missing: {expected - present}"
    for p in split_dir.glob("*.ipynb"):
        nb = nbf.read(p, as_version=4)          # raises on malformed JSON/schema
        assert len(nb.cells) > 5, p.name


def test_setup_cells_shared_across_phases(gen):
    """Each phase embeds the same install/env/model-load preamble."""
    for name, build in _all_phase_builders(gen).items():
        srcs = "\n".join(_code_sources(build()))
        assert "smart_load_model()" in srcs, name        # model load
        assert "PYTORCH_CUDA_ALLOC_CONF" in srcs, name    # cuda alloc preamble
