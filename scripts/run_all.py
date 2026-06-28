#!/usr/bin/env python3
"""Headless end-to-end driver — runs the whole pipeline on ONE GPU.

This is the server-side equivalent of the notebooks: it runs discovery
(H1-H5), the H6 benchmark + consensus-flip, the MedQA-scale calibration
(H7), the H7-informed causal re-test (H6 pass-2), the cross-task split
(H8), and the sycophancy probe — sequentially, no notebook, no multi-GPU
spawn (a single H200 MIG slice is one CUDA device).

Every stage is wrapped so a failure logs + continues to the next stage, and
every stage is resumable (it reloads its own JSON/JSONL if present). Re-running
after an interruption picks up where it left off.

Tunables (env):
  N_BENCH     H6/H6-pass2 question count           (default 1273 = full MedQA test)
  H7_N        H7 answer-position collection count   (default 300)
  H8_N        H8 cross-task pair count              (default 200)
  SYC_N       sycophancy probe count                (default 300)
  DP_MAX_NEW_TOKENS   generation cap                (default 256; early-stop usually halts sooner)
  RESULTS_DIR override the results directory        (default <repo>/results)

Run:  python scripts/run_all.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — never try to open a display

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)

N_BENCH = int(os.environ.get("N_BENCH", "1273"))
H7_N = int(os.environ.get("H7_N", "300"))
H8_N = int(os.environ.get("H8_N", "200"))
SYC_N = int(os.environ.get("SYC_N", "300"))
DATASET = os.environ.get("DP_DATASET", "GBaker/MedQA-USMLE-4-options-hf")

_t_start = time.time()


def banner(msg: str) -> None:
    el = time.time() - _t_start
    print("\n" + "=" * 78)
    print(f"  {msg}   [+{el/60:.1f} min]")
    print("=" * 78, flush=True)


def stage(name: str):
    """Decorator: run a stage, catch+log exceptions, return ok flag."""
    def deco(fn):
        def wrapped(*a, **k):
            banner(name)
            try:
                fn(*a, **k)
                print(f"[ok] {name}", flush=True)
                return True
            except Exception:
                print(f"[FAIL] {name} — continuing to next stage:", flush=True)
                traceback.print_exc()
                return False
        return wrapped
    return deco


# Shared state populated as stages run.
S: dict = {}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load():
    banner("Loading model")
    from src.setup import auto_pick, smart_load_model
    plan = auto_pick()
    print("Plan:", {k: plan[k] for k in ("model", "use_4bit", "n_gpus", "gpu_gb")})
    lm, model_name = smart_load_model(plan)
    S["lm"] = lm
    S["MODEL_NAME"] = model_name
    import torch
    S["sha"] = os.environ.get("DP_GIT_SHA", "")
    print(f">>> Loaded {model_name}: layers={lm.n_layers} d_ff={lm.d_ff} "
          f"dtype={lm.dtype} device={lm.device}")
    if torch.cuda.is_available():
        print(f"    VRAM allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")


def free_vram(tag=""):
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  [vram {tag}] {torch.cuda.memory_allocated()/1e9:.1f} GB allocated", flush=True)


# ---------------------------------------------------------------------------
# Discovery (H1-H5)
# ---------------------------------------------------------------------------


@stage("H1 — diagnosis-gate discovery + capability-aware multiplier")
def h1():
    import torch
    from types import SimpleNamespace
    from src.data import build_h1
    from src.discover import (
        discover, sweep, best_multiplier_with_capability,
        mean_target_logprob_under, _target_first_token_ids, _mean_target_logprob,
        DEFAULT_M_SWEEP, NeuronScore,
    )
    lm = S["lm"]
    H1 = RESULTS / "h1"; H1.mkdir(exist_ok=True)
    h1d = build_h1()

    cp = H1 / "candidates.json"
    if cp.exists():
        cands = [NeuronScore(**c) for c in json.loads(cp.read_text())]
    else:
        cands = discover(lm, positive_prompts=h1d["positive"],
                         negative_prompts=h1d["negative"],
                         target_phrases=h1d["commitment_phrases"],
                         icd10_tokens=h1d["icd10_tokens"], layer_range=None, top_k=5)
        cp.write_text(json.dumps([c.__dict__ for c in cands], indent=2))
    print("Top gate candidates:", [(f"L{c.layer}:F{c.neuron}", round(c.score, 3)) for c in cands])

    probe = h1d["positive"][:8]
    hard = h1d["hard_cases"][:8]
    sw = sweep(lm, candidates=cands, probes=probe,
               target_phrases=h1d["commitment_phrases"], icd10_tokens=h1d["icd10_tokens"],
               multipliers=DEFAULT_M_SWEEP, sample_prompt=h1d["positive"][0])
    (H1 / "sweep.json").write_text(json.dumps([s.__dict__ for s in sw], indent=2))
    cap = mean_target_logprob_under(lm, candidates=cands, capability_prompts=hard,
                                    target_phrases=h1d["commitment_phrases"],
                                    icd10_tokens=h1d["icd10_tokens"], multipliers=DEFAULT_M_SWEEP)
    (H1 / "capability_sweep.json").write_text(json.dumps({str(k): v for k, v in cap.items()}, indent=2))
    tids = _target_first_token_ids(lm.tokenizer, h1d["commitment_phrases"], h1d["icd10_tokens"])
    base_cap = _mean_target_logprob(lm, hard, tids)

    L_star, N_star, m_star, _ = best_multiplier_with_capability(
        [SimpleNamespace(**s.__dict__) for s in sw], cap, base_cap, lambda_cap=1.0)
    best = next(c for c in cands if c.layer == L_star and c.neuron == N_star)
    anchor_d = float(best.a_pos - best.a_neg) or 1e-3
    print(f"Gate: L{L_star}:F{N_star} m*={m_star} anchor_d={anchor_d:.4f}")
    S.update(cands=cands, L_star=L_star, N_star=N_star, m_star=m_star, anchor_d=anchor_d)
    free_vram("after H1")


@stage("H2 — disease concept neurons + amplification")
def h2():
    from src.concept import DISEASE_KEYWORDS, amplification_matrix, rank_concept_neurons
    from src.data import build_h2
    lm = S["lm"]
    H2 = RESULTS / "h2"; H2.mkdir(exist_ok=True)
    h2d = build_h2(n_per_disease=200)
    cn = {}
    live = {}
    for disease in [k for k in h2d if not k.startswith("_")]:
        pos = h2d[disease]["positive"]; neg = h2d[disease]["negative"]
        neurons = rank_concept_neurons(lm, positive=pos, negative=neg, top_k=3, disease=disease)
        cn[disease] = [n.__dict__ for n in neurons]
        live[disease] = neurons
    (H2 / "concept_neurons.json").write_text(json.dumps(cn, indent=2))
    print("Concept neurons per disease:", {d: len(v) for d, v in cn.items()})
    # Light amplification check on the top neuron of the first two diseases —
    # pass the live ConceptNeuron objects straight through (no reconstruction).
    benign = h2d["_benign_prompts"]["positive"][:4]
    amp = {}
    for disease in list(live)[:2]:
        if not live[disease]:
            continue
        rows = amplification_matrix(lm, neuron=live[disease][0], benign_prompts=benign,
                                    multipliers=[0.0, 1.0, 2.0, 4.0, 8.0],
                                    max_new_tokens=64,
                                    concept_keywords=DISEASE_KEYWORDS[disease])
        amp[disease] = [r.__dict__ for r in rows]
    (H2 / "amplification.json").write_text(json.dumps(amp, indent=2))
    free_vram("after H2")


@stage("H3 — symptom→diagnosis routing (patching)")
def h3():
    import numpy as np
    from src.data import H3_PAIRS, verify_h3_tokens
    from src.patching import patch_layers, patch_neurons_at_layer
    lm = S["lm"]
    H3 = RESULTS / "h3"; H3.mkdir(exist_ok=True)
    verify_h3_tokens(lm.tokenizer)
    patch_data = {}
    for pair in H3_PAIRS:
        scores = patch_layers(lm, pair.clean_prompt, pair.corrupted_prompt,
                              pair.clean_dx, pair.corrupted_dx, pair.pair_id)
        patch_data[pair.pair_id] = [s.__dict__ for s in scores]
    (H3 / "patch_layers.json").write_text(json.dumps(patch_data, indent=2))
    # Critical layer = max mean score across pairs.
    by_layer = {}
    for recs in patch_data.values():
        for r in recs:
            by_layer.setdefault(r["layer"], []).append(r["score"])
    mean_per_layer = {L: float(np.mean(v)) for L, v in by_layer.items()}
    critical = max(mean_per_layer, key=mean_per_layer.get)
    print(f"Critical layer L{critical} (mean {mean_per_layer[critical]:+.3f})")
    S.update(critical=critical, mean_per_layer=mean_per_layer)
    # Drill the critical layer on the first pair (diagnostic; bounded).
    pair = H3_PAIRS[0]
    drill = patch_neurons_at_layer(lm, pair.clean_prompt, pair.corrupted_prompt,
                                   pair.clean_dx, pair.corrupted_dx, layer_idx=critical,
                                   neuron_indices=list(range(lm.d_ff)), pair_id=pair.pair_id)
    (H3 / f"drill_L{critical}_full.json").write_text(
        json.dumps([d.__dict__ for d in drill], indent=2))
    free_vram("after H3")


@stage("H4 — hallucination / false-confidence neurons")
def h4():
    from src.data import build_h1, build_h4
    from src.hallucinate import find_hallucination_neurons
    lm = S["lm"]
    H4 = RESULTS / "h4"; H4.mkdir(exist_ok=True)
    h4d = build_h4(); h1d = build_h1()
    halluc, classif = find_hallucination_neurons(
        lm, trap_prompts=h4d["trap"], pathognomonic_prompts=h4d["pathognomonic"][:10],
        hedge_prompts=h1d["negative"][:10], target_phrases=h4d["commitment_phrases"],
        icd10_tokens=h4d["icd10_tokens"], layer_range=None, top_k=10, commit_p_threshold=0.10)
    (H4 / "hallucination_neurons.json").write_text(json.dumps([n.__dict__ for n in halluc], indent=2))
    (H4 / "classifications.json").write_text(json.dumps(classif, indent=2))
    n_trap = len(classif.get("trap", []))
    committed = sum(1 for _, c, _ in classif.get("trap", []) if c)
    print(f"Trap commit rate: {committed}/{n_trap}")
    S["halluc_neurons"] = halluc
    free_vram("after H4")


@stage("H5 — overconfidence / miscalibration neurons")
def h5():
    from src.calibration import find_overconfidence_neurons
    from src.data import build_h1
    lm = S["lm"]
    H5 = RESULTS / "h5"; H5.mkdir(exist_ok=True)
    h1d = build_h1()
    cases, over = find_overconfidence_neurons(
        lm, hard_cases=h1d["hard_cases"], layer_range=None,
        top_k=15, overconf_threshold=0.3, gap_high_low_n=4)
    dump = [{"case": c.case, "dx_text": c.dx_text, "p_dx": c.p_dx, "p_yes": c.p_yes,
             "p_no": c.p_no, "calibration_gap": c.calibration_gap} for c in cases]
    (H5 / "cases.json").write_text(json.dumps(dump, indent=2))
    (H5 / "overconfidence_neurons.json").write_text(json.dumps([n.__dict__ for n in over], indent=2))
    print(f"H5: {len(cases)} cases; top neuron r={over[0].pearson_r:+.3f}" if over else "H5: no neurons")
    S["over_neurons"] = over
    free_vram("after H5")


@stage("Save discovery checkpoint")
def save_discovery():
    from src.checkpoint import Discovery, save_discovery as save_disc
    lm = S["lm"]
    # Defensive: only write if the H1 core (gate + critical layer) succeeded.
    # Optional neuron lists default to [] so a partial discovery still persists
    # the valid gate rather than crashing on the first missing key.
    core = ("L_star", "N_star", "m_star", "anchor_d", "critical")
    missing = [k for k in core if k not in S]
    if missing:
        print(f"discovery incomplete (missing {missing}) — not writing checkpoint.")
        return
    mpl = S.get("mean_per_layer", {})
    disc = Discovery(
        model_name=S.get("MODEL_NAME", "?"),
        gate_layer=int(S["L_star"]), gate_neuron=int(S["N_star"]),
        gate_m_star=float(S["m_star"]), gate_anchor_d=float(S["anchor_d"]),
        critical_layer=int(S["critical"]),
        layer_scores=[float(mpl[k]) for k in sorted(mpl)],
        halluc_neurons=[{"layer": int(n.layer), "neuron": int(n.neuron)}
                        for n in S.get("halluc_neurons", [])[:3]],
        overconf_neurons=[{"layer": int(n.layer), "neuron": int(n.neuron)}
                          for n in S.get("over_neurons", [])[:3]],
        git_sha=S.get("sha", ""),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_layers=int(lm.n_layers), d_ff=int(lm.d_ff),
        extra={"mean_per_layer": {str(k): float(v) for k, v in mpl.items()}},
    )
    save_disc(RESULTS, disc)
    print("Wrote", RESULTS / "discovery.json")


# ---------------------------------------------------------------------------
# Fault isolation: post-discovery stages reload what they need from disk so a
# single upstream failure doesn't KeyError-cascade through the whole pipeline.
# ---------------------------------------------------------------------------


def _ensure_items():
    if "items" not in S:
        from src.healthbench import load_medqa
        S["items"] = load_medqa(DATASET, split="test", n=N_BENCH, seed=0)
        print(f"  (reloaded {len(S['items'])} items)")
    return S["items"]


def _ensure_discovery_coords():
    """Populate gate/critical/neuron coords in S from results/discovery.json if
    an upstream discovery stage failed to set them in this process."""
    from types import SimpleNamespace
    needed = ("L_star", "N_star", "m_star", "anchor_d", "critical",
              "over_neurons", "halluc_neurons", "mean_per_layer")
    if all(k in S for k in needed):
        return
    from src.checkpoint import load_discovery   # raises a clear error if 01 never ran
    disc = load_discovery(RESULTS)
    S.setdefault("L_star", disc.gate_layer)
    S.setdefault("N_star", disc.gate_neuron)
    S.setdefault("m_star", disc.gate_m_star)
    S.setdefault("anchor_d", disc.gate_anchor_d)
    S.setdefault("critical", disc.critical_layer)
    S.setdefault("over_neurons", [SimpleNamespace(**d) for d in disc.overconf_neurons])
    S.setdefault("halluc_neurons", [SimpleNamespace(**d) for d in disc.halluc_neurons])
    mpl = (disc.extra or {}).get("mean_per_layer", {})
    S.setdefault("mean_per_layer", {int(k): float(v) for k, v in mpl.items()})
    print("  (restored discovery coords from results/discovery.json)")


# ---------------------------------------------------------------------------
# H6 benchmark + consensus
# ---------------------------------------------------------------------------


def _build_conditions():
    from src.healthbench import (
        ablate_neurons_factory, anchor_factory, zero_mlp_factory,
    )
    lm = S["lm"]
    top_overconf = [{"layer": n.layer, "neuron": n.neuron} for n in S["over_neurons"][:3]]
    top_halluc = [{"layer": n.layer, "neuron": n.neuron} for n in S["halluc_neurons"][:3]]
    S.update(top_overconf=top_overconf, top_halluc=top_halluc,
             combined=top_overconf + top_halluc)
    return {
        "baseline": None,
        "h1_gate_anchor": anchor_factory(lm.layers, S["L_star"], S["N_star"],
                                         S["m_star"], S["anchor_d"], k=1.0),
        "h3_zero_layer": zero_mlp_factory(lm.layers, [S["critical"]]),
        "h4_ablate_halluc": ablate_neurons_factory(lm.layers, top_halluc),
        "h5_ablate_overconf": ablate_neurons_factory(lm.layers, top_overconf),
        "h4_h5_combined": ablate_neurons_factory(lm.layers, S["combined"]),
    }


@stage("H6 — benchmark under interventions + consensus-flip")
def h6():
    from src.healthbench import run_conditions
    from src.consensus import analyze, summarize
    lm = S["lm"]
    H6 = RESULTS / "h6"; H6.mkdir(exist_ok=True)
    _ensure_discovery_coords()      # tolerate an upstream discovery failure
    items = _ensure_items()
    print(f"Loaded {len(items)} questions.")
    conds = _build_conditions()
    S["CONDITIONS"] = conds
    run_conditions(lm, items, conds, out_dir=H6, save_every=25)
    summary = json.loads((H6 / "summary.json").read_text())
    for c, s in summary.items():
        print(f"  {c:<22} acc={s['accuracy']:.3f} brier={s['brier_at_answer']:.4f}")
    rows = analyze(H6 / "comparison.csv", list(conds.keys()))
    rep = summarize(rows, list(conds.keys()))
    (H6 / "consensus_flip.json").write_text(
        json.dumps({"report": rep, "rows": [r.__dict__ for r in rows]}, indent=2))
    print(f"Consensus-flips: {rep['n_consensus_flips']}/{rep['n_total']}")
    free_vram("after H6")


# ---------------------------------------------------------------------------
# H7 + H6 pass-2 + H8
# ---------------------------------------------------------------------------


@stage("H7 — MedQA-scale miscalibration layers")
def h7():
    from src.h7_layers import collect_answer_position_acts, rank_miscalibration_neurons
    lm = S["lm"]
    H7 = RESULTS / "h7"; H7.mkdir(exist_ok=True)
    items = _ensure_items()[:min(H7_N, len(_ensure_items()))]
    rows, acts = collect_answer_position_acts(
        lm, items, layer_indices=list(range(lm.n_layers // 2, lm.n_layers)))
    S["h7_rows"], S["h7_acts"] = rows, acts
    neurons = rank_miscalibration_neurons(rows, acts, top_k=20)
    S["miscal_neurons"] = neurons
    (H7 / "miscal_neurons.json").write_text(json.dumps([n.__dict__ for n in neurons], indent=2))
    (H7 / "rows.json").write_text(json.dumps(rows, indent=2))
    print(f"H7: {len(rows)} rows, {len(neurons)} neurons pass FDR; "
          f"top r={neurons[0].pearson_r:+.3f}" if neurons else "H7: none pass FDR")


@stage("H6 pass-2 — H7-informed causal re-test")
def h6_pass2():
    from src.healthbench import additive_shift_factory, ablate_neurons_factory, run_conditions
    lm = S["lm"]
    H6 = RESULTS / "h6"
    mn = S.get("miscal_neurons") or []
    if not mn:
        print("No H7 neurons — skipping pass-2.")
        return
    _ensure_discovery_coords()
    items = _ensure_items()
    top_h7 = [{"layer": n.layer, "neuron": n.neuron} for n in mn[:3]]
    shifts = [{"layer": n.layer, "neuron": n.neuron,
               "amount": float(n.mean_act_calib - n.mean_act_overconf)} for n in mn[:3]]
    # Rebuild the base conditions if h6() didn't run in this process.
    conds = dict(S.get("CONDITIONS") or _build_conditions())
    conds["h7_ablate_miscal"] = ablate_neurons_factory(lm.layers, top_h7)
    conds["h7_anchor_calibrated"] = additive_shift_factory(lm.layers, shifts)
    run_conditions(lm, items, conds, out_dir=H6, save_every=25)
    summary = json.loads((H6 / "summary.json").read_text())
    base = summary["baseline"]["brier_at_answer"]
    for c in ("h5_ablate_overconf", "h7_ablate_miscal", "h7_anchor_calibrated"):
        if c in summary:
            print(f"  {c:<22} dBrier={summary[c]['brier_at_answer']-base:+.4f}")
    free_vram("after H6 pass-2")


@stage("H8 — cross-task confidence circuits")
def h8():
    from src.h8_xtask import collect_xtask, classify_neurons
    lm = S["lm"]
    H8 = RESULTS / "h8"; H8.mkdir(exist_ok=True)
    items = _ensure_items()[:min(H8_N, len(_ensure_items()))]
    rows, acts_mcq, acts_prose = collect_xtask(
        lm, items, layer_indices=list(range(lm.n_layers // 2, lm.n_layers)))
    neurons = classify_neurons(rows, acts_mcq, acts_prose, r_threshold=0.15)
    (H8 / "rows.json").write_text(json.dumps([r.__dict__ for r in rows], indent=2))
    (H8 / "neurons.json").write_text(json.dumps([n.__dict__ for n in neurons], indent=2))
    cats = {}
    for n in neurons:
        cats[n.category] = cats.get(n.category, 0) + 1
    print("H8 categories:", cats)
    free_vram("after H8")


# ---------------------------------------------------------------------------
# Sycophancy
# ---------------------------------------------------------------------------


@stage("Sycophancy — probe + neurons + reduction")
def sycophancy():
    from src.sycophancy import (
        run_sycophancy_probe, summarize_probe, find_sycophancy_neurons,
    )
    lm = S["lm"]
    SY = RESULTS / "sycophancy"; SY.mkdir(exist_ok=True)
    items = _ensure_items()[:min(SYC_N, len(_ensure_items()))]
    cases = run_sycophancy_probe(lm, items, n_questions=len(items), seed=0)
    summ = summarize_probe(cases)
    (SY / "cases.json").write_text(json.dumps([c.__dict__ for c in cases], indent=2))
    (SY / "summary.json").write_text(json.dumps(summ, indent=2))
    print("Sycophancy:", {k: summ[k] for k in
                          ("baseline_accuracy", "authority_flip_to_user",
                           "insistence_flip_to_user") if k in summ})
    items_by_qid = {it.q_id: it for it in items}
    try:
        neurons = find_sycophancy_neurons(lm, cases, items_by_qid, top_k=20)
        (SY / "neurons.json").write_text(json.dumps([n.__dict__ for n in neurons], indent=2))
        print(f"Sycophancy neurons: {len(neurons)}")
    except RuntimeError as e:
        print("No sycophancy-flip cases to analyze:", e)
    free_vram("after sycophancy")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    load()
    h1(); h2(); h3(); h4(); h5(); save_discovery()
    h6()
    h7(); h6_pass2(); h8()
    sycophancy()
    banner(f"PIPELINE DONE — results in {RESULTS}")
    print("Total wall time: %.1f min" % ((time.time() - _t_start) / 60))


if __name__ == "__main__":
    main()
