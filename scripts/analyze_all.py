#!/usr/bin/env python3
"""Level-by-level analysis of a completed (or partial) results/ folder.

Self-contained: stdlib + json/csv only (matplotlib optional for plots). Walks
the pipeline H1 -> H2 -> H3 -> H4 -> H5 -> H6 -> H7 -> H8 -> sycophancy, computes
the headline metrics + statistics for each, flags methodological problems
(e.g. saturated H3 patching), and writes:

    <results>/analysis/report.md     human-readable level-by-level report
    <results>/analysis/report.json   machine-readable metrics
    <results>/analysis/*.png         a few plots (if matplotlib present)

Missing phases are skipped with a note — so it works on a partial run too.

Run:  python scripts/analyze_all.py            (uses ./results)
      python scripts/analyze_all.py /path/to/results
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = RESULTS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

_md: List[str] = []
_json: Dict = {}


def w(line: str = "") -> None:
    _md.append(line)
    print(line)


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def sec_discovery():
    d = _load(RESULTS / "discovery.json")
    w("## 0. Discovery checkpoint")
    if not d:
        w("_no discovery.json — discovery (phase 01) did not complete._\n")
        return
    _json["discovery"] = d
    w(f"- **Model**: `{d.get('model_name')}`  ({d.get('n_layers')} layers, d_ff={d.get('d_ff')})")
    w(f"- **H1 gate**: L{d.get('gate_layer')}:F{d.get('gate_neuron')}  "
      f"(m\\*={d.get('gate_m_star')}, anchor_d={d.get('gate_anchor_d', 0.0):.3f})")
    w(f"- **H3 critical layer**: L{d.get('critical_layer')}")
    w(f"- **H4 hallucination neurons**: {d.get('halluc_neurons')}")
    w(f"- **H5 overconfidence neurons**: {d.get('overconf_neurons')}")
    w(f"- discovery git sha `{d.get('git_sha','?')}`, created {d.get('created_utc','?')}\n")


# ---------------------------------------------------------------------------
# H1 / H2
# ---------------------------------------------------------------------------


def sec_h1():
    w("## H1 — diagnosis-gate neuron")
    cand = _load(RESULTS / "h1" / "candidates.json")
    sweep = _load(RESULTS / "h1" / "sweep.json")
    cap = _load(RESULTS / "h1" / "capability.json")
    if not cand:
        w("_no H1 outputs._\n"); return
    w("Top-5 gate candidates (gradient × activation, paper Eq. 4):\n")
    w("| neuron | score | a_pos | a_neg |")
    w("|---|---|---|---|")
    for c in cand[:5]:
        w(f"| L{c['layer']}:F{c['neuron']} | {c['score']:+.3f} | {c['a_pos']:+.2f} | {c['a_neg']:+.2f} |")
    if cap:
        hedge = lambda k: sum(int(r.get(k, False)) for r in cap) / max(1, len(cap))
        w(f"\nHard-case hedge rate — baseline {hedge('baseline_hedge'):.2f}, "
          f"constant {hedge('constant_hedge'):.2f}, anchor {hedge('anchor_hedge'):.2f} "
          f"(anchor > baseline ⇒ suppressing the gate pushes the model toward hedging).")
    _json["h1"] = {"n_candidates": len(cand)}
    w("")


def sec_h2():
    cn = _load(RESULTS / "h2" / "concept_neurons.json")
    w("## H2 — disease concept neurons")
    if not cn:
        w("_no H2 outputs._\n"); return
    w("| disease | top neuron | margin |")
    w("|---|---|---|")
    for dis, neurons in cn.items():
        if neurons:
            n = neurons[0]
            margin = n.get("margin", n.get("score", n.get("cohens_d", "")))
            ms = f"{margin:+.2f}" if isinstance(margin, (int, float)) else str(margin)
            w(f"| {dis} | L{n.get('layer')}:F{n.get('neuron')} | {ms} |")
    _json["h2"] = {"diseases": list(cn.keys())}
    w("")


# ---------------------------------------------------------------------------
# H3 — with the saturation artifact check
# ---------------------------------------------------------------------------


def sec_h3():
    pl = _load(RESULTS / "h3" / "patch_layers.json")
    w("## H3 — symptom→diagnosis routing (activation patching)")
    if not pl:
        w("_no H3 outputs._\n"); return
    by_layer: Dict[int, List[float]] = {}
    for recs in pl.values():
        for r in recs:
            by_layer.setdefault(r["layer"], []).append(r["score"])
    means = {L: sum(v) / len(v) for L, v in by_layer.items()}
    layers = sorted(means)
    vals = [means[L] for L in layers]
    saturated = all(abs(v - 1.0) < 1e-6 for v in vals)
    spread = (max(vals) - min(vals)) if vals else 0.0
    _json["h3"] = {"n_layers": len(layers), "score_spread": spread, "saturated": saturated}
    if saturated:
        w("> ⚠️ **BROKEN / uninterpretable.** Patch score = 1.000 at *every* layer "
          f"({len(layers)} layers). That means patching any single layer fully "
          "recovers the clean answer — the metric is copying the whole clean "
          "residual, not localizing routing. The 'critical layer' is an artifact; "
          "do not report an H3 routing result until the patching gives a "
          "non-saturated curve (the token-matched pairs still aren't isolating a "
          "single layer).\n")
    else:
        top = sorted(means.items(), key=lambda t: t[1], reverse=True)[:3]
        w(f"Per-layer mean patch score spread = {spread:.3f}. "
          f"Top layers: {[(f'L{L}', round(s,2)) for L,s in top]}\n")


# ---------------------------------------------------------------------------
# H4 / H5
# ---------------------------------------------------------------------------


def sec_h4():
    cl = _load(RESULTS / "h4" / "classifications.json")
    w("## H4 — hallucination / false confidence")
    if not cl:
        w("_no H4 outputs._\n"); return
    def rate(label, want=True):
        L = cl.get(label, [])
        c = sum(1 for _, com, _ in L if bool(com) == want)
        return c, len(L)
    tc, tn = rate("trap")
    pc, pn = rate("pathognomonic")
    w(f"- **Trap prompts committed (hallucinated a diagnosis): {tc}/{tn} "
      f"({100*tc/max(1,tn):.0f}%)** — these include underspecified, contradictory, "
      "fabricated and biologically-impossible cases a clinician would refuse.")
    w(f"- Pathognomonic committed (should be high): {pc}/{pn} ({100*pc/max(1,pn):.0f}%)")
    if tn and tc == tn:
        w("- **Zero refusal**: the model committed on 100% of traps — a clean "
          "false-confidence signal.")
    _json["h4"] = {"trap_commit": tc, "trap_n": tn}
    w("")


def sec_h5():
    neurons = _load(RESULTS / "h5" / "overconfidence_neurons.json")
    cases = _load(RESULTS / "h5" / "cases.json")
    w("## H5 — overconfidence neurons (discovery scale, small N)")
    if not neurons:
        w("_no H5 outputs._\n"); return
    n = neurons[0]
    # NB: keep this f-string un-nested — same-quote nested f-strings are a
    # SyntaxError before Python 3.12 and the server may run 3.10/3.11.
    top3 = [(f"L{x['layer']}:F{x['neuron']}", round(x["pearson_r"], 2)) for x in neurons[:3]]
    w(f"- Top neurons by Pearson r(activation, calibration-gap): {top3}")
    w(f"- Top r = **{n['pearson_r']:+.2f}** at n={n.get('n_cases','?')} cases.")
    if cases:
        gaps = [c["calibration_gap"] for c in cases]
        w(f"- Calibration gap (p_yes − p_dx) over {len(cases)} hard cases: "
          f"mean {sum(gaps)/len(gaps):+.3f}, max {max(gaps):+.3f}.")
    w("> ⚠️ Effect sizes this large (r≈0.85) at n≈20 are exactly the small-N regime "
      "where correlations inflate. **H7 is the scale-up (n≥300 + FDR) that confirms "
      "or refutes these** — trust H7 over H5 for the causal claim.\n")
    _json["h5"] = {"top_r": n["pearson_r"], "n_cases": n.get("n_cases")}


# ---------------------------------------------------------------------------
# H6 — the heart of it
# ---------------------------------------------------------------------------


def _ece_and_bins(rows, cond):
    bins = [[] for _ in range(10)]
    used = 0
    for r in rows:
        if r.get(f"{cond}_answer_found") != "1":
            continue
        p = _f(r.get(f"{cond}_p_top1_answer"))
        c = r.get(f"{cond}_correct")
        if p is None or c not in ("0", "1"):
            continue
        bins[min(9, int(p * 10))].append((p, int(c)))
        used += 1
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(c for _, c in b) / len(b)
        ece += len(b) / max(1, used) * abs(conf - acc)
    return ece, used


def sec_h6():
    w("## H6 — benchmark under interventions  (the causal test)")
    summ = _load(RESULTS / "h6" / "summary.json")
    comp = RESULTS / "h6" / "comparison.csv"
    if not summ:
        w("_no H6 summary._\n"); return
    rows = list(csv.DictReader(open(comp))) if comp.exists() else []
    n_done = {c: 0 for c in summ}
    if rows:
        for c in summ:
            n_done[c] = sum(1 for r in rows if r.get(f"{c}_pred", "") != "")
    w(f"Questions: {summ.get('baseline',{}).get('n','?')}  "
      f"(comparison.csv rows: {len(rows)})\n")
    w("| condition | n | acc | mean p_top1@ans | brier@ans | ECE | conf>0.9 & wrong |")
    w("|---|---|---|---|---|---|---|")
    base_acc = summ.get("baseline", {}).get("accuracy")
    base_brier = summ.get("baseline", {}).get("brier_at_answer")
    h6j = {}
    # Only list baseline first if it actually exists (partial runs may not have it).
    order = (["baseline"] if "baseline" in summ else []) + [c for c in summ if c != "baseline"]
    for c in order:
        s = summ[c]
        if rows:
            ece, used = _ece_and_bins(rows, c)
            af = sum(1 for r in rows if r.get(f"{c}_answer_found") == "1")
            cw = sum(1 for r in rows if r.get(f"{c}_answer_found") == "1"
                     and r.get(f"{c}_correct") == "0"
                     and (_f(r.get(f"{c}_p_top1_answer")) or 0) > 0.9)
            cw_rate = 100 * cw / max(1, af)
            ece_s, cw_s = f"{ece:.3f}", f"{cw_rate:.1f}%"
        else:
            ece, cw_rate = None, None     # JSON-valid (null), not NaN
            ece_s, cw_s = "n/a", "n/a"
        w(f"| {c} | {s['n']} | {s['accuracy']:.3f} | {s['mean_p_top1_at_answer']:.3f} "
          f"| {s['brier_at_answer']:.4f} | {ece_s} | {cw_s} |")
        h6j[c] = {"acc": s["accuracy"], "brier": s["brier_at_answer"], "ece": ece,
                  "confidently_wrong_pct": cw_rate}
    _json["h6"] = h6j

    # Reliability of the baseline.
    if rows:
        w("\n**Baseline reliability** (binned by answer-position confidence):\n")
        w("| confidence | n | share | accuracy | over-confidence |")
        w("|---|---|---|---|---|")
        bands = [("≥0.99", 0.99, 1.01, 0.995), ("0.95–0.99", 0.95, 0.99, 0.97),
                 ("0.90–0.95", 0.90, 0.95, 0.925), ("<0.90", 0.0, 0.90, 0.7)]
        af = [r for r in rows if r.get("baseline_answer_found") == "1"]
        for name, lo, hi, mid in bands:
            sub = [r for r in af if lo <= (_f(r.get("baseline_p_top1_answer")) or -1) < hi]
            if not sub:
                continue
            acc = sum(1 for r in sub if r.get("baseline_correct") == "1") / len(sub)
            w(f"| {name} | {len(sub)} | {100*len(sub)/len(af):.1f}% | {acc:.3f} | {mid-acc:+.3f} |")

    # Intervention verdict.
    if base_acc is not None:
        w("\n**Causal effect of the interventions** (vs baseline):\n")
        moved = False
        for c in order:
            if c == "baseline":
                continue
            dacc = summ[c]["accuracy"] - base_acc
            dbri = summ[c]["brier_at_answer"] - base_brier
            tag = ""
            if c == "h3_zero_layer" and summ[c]["accuracy"] < base_acc - 0.2:
                tag = " ← destroys the model (layer-0 is load-bearing, not a routing result)"
            # Exclude the model-destroyer from the "did anything move?" test —
            # it always moves accuracy by a lot, which would wrongly suppress
            # the null-result paragraph (must match the main() verdict logic).
            if c != "h3_zero_layer" and abs(dacc) >= 0.02:
                moved = True
            w(f"- {c}: Δacc {dacc:+.3f}, Δbrier {dbri:+.4f}{tag}")
        if not moved:
            w("\n> **Key result:** the single-neuron interventions do **not** "
              "meaningfully move accuracy or calibration at scale. The strong "
              "discovery-phase correlations (H1/H5) do not translate into a causal "
              "effect on the benchmark — this *argues against* the 'a single neuron "
              "is sufficient' thesis transferring to clinical calibration. (Caveat: "
              "weak m\\* and only 3 neurons ablated; try stronger interventions "
              "before concluding.)")
    w("")


def sec_consensus():
    cf = _load(RESULTS / "h6" / "consensus_flip.json")
    w("## H6b — consensus-flip (knows-but-says-wrong)")
    if not cf:
        w("_no consensus_flip.json._\n"); return
    r = cf["report"]
    w(f"- consensus-flip cases: **{r['n_consensus_flips']}** of {r['n_baseline_wrong']} "
      f"baseline-wrong ({r['n_total']} total)")
    w("\n| condition | fixed on flips | on-flips % | on-any-wrong % |")
    w("|---|---|---|---|")
    for c, info in r["fix_rates"].items():
        w(f"| {c} | {info['on_flips']} | {100*info['on_flips_rate']:.1f}% | "
          f"{100*info['on_any_rate']:.1f}% |")
    if r["n_consensus_flips"] < 25:
        w(f"\n> Underpowered: only {r['n_consensus_flips']} flip cases. Enrichment "
          "ratios are suggestive but rest on single-digit counts.")
    _json["consensus"] = {"n_flips": r["n_consensus_flips"]}
    w("")


# ---------------------------------------------------------------------------
# H7 / H8 / sycophancy
# ---------------------------------------------------------------------------


def sec_h7():
    mn = _load(RESULTS / "h7" / "miscal_neurons.json")
    w("## H7 — MedQA-scale miscalibration layers (FDR)")
    if mn is None:
        w("_not run (phase 03)._ This is the most important missing piece — it is "
          "the FDR-corrected, n≥300 confirmation of the H5 overconfidence neurons.\n")
        return
    if not mn:
        w("0 neurons survived BH-FDR — the H5 discovery signal did **not** replicate "
          "at scale (it was small-N overfitting).\n")
        _json["h7"] = {"n_fdr": 0}
        return
    disc = _load(RESULTS / "discovery.json") or {}
    h5set = {(x["layer"], x["neuron"]) for x in disc.get("overconf_neurons", [])}
    h7set = {(n["layer"], n["neuron"]) for n in mn}
    overlap = h5set & h7set
    w(f"- {len(mn)} neurons pass FDR; top r = {mn[0]['pearson_r']:+.3f}")
    w(f"- overlap with H5 top-3: {len(overlap)} {sorted(overlap)}")
    _json["h7"] = {"n_fdr": len(mn), "top_r": mn[0]["pearson_r"], "h5_overlap": len(overlap)}
    w("")


def sec_h8():
    nn = _load(RESULTS / "h8" / "neurons.json")
    w("## H8 — cross-task confidence circuits")
    if nn is None:
        w("_not run (phase 03)._\n"); return
    cats = {}
    for n in nn:
        cats[n.get("category")] = cats.get(n.get("category"), 0) + 1
    w(f"- neuron categories: {cats}")
    gen = cats.get("general", 0)
    w(f"- task-general 'I'm sure' neurons (fire in both MCQ & prose): {gen}")
    _json["h8"] = cats
    w("")


def sec_syc():
    s = _load(RESULTS / "sycophancy" / "summary.json")
    w("## Sycophancy — does the model cave to a wrong user claim?")
    if not s:
        w("_not run (phase 04)._\n"); return
    w(f"- baseline accuracy on probed subset: {s.get('baseline_accuracy',0):.3f}")
    w(f"- **authority push** flip-to-user rate: {s.get('authority_flip_to_user',0):.3f}")
    w(f"- **insistence push** flip-to-user rate: {s.get('insistence_flip_to_user',0):.3f}")
    w(f"- correct→wrong under authority: {s.get('authority_correct_to_wrong_rate',0):.3f}, "
      f"insistence: {s.get('insistence_correct_to_wrong_rate',0):.3f}")
    _json["sycophancy"] = {k: s.get(k) for k in
                           ("authority_flip_to_user", "insistence_flip_to_user")}
    w("")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    w(f"# Diagnostic Percept — analysis report")
    w(f"_results: `{RESULTS}`_\n")
    sec_discovery()
    sec_h1(); sec_h2(); sec_h3(); sec_h4(); sec_h5()
    sec_h6(); sec_consensus()
    sec_h7(); sec_h8(); sec_syc()

    # Overall verdict.
    w("## Verdict")
    h6 = _json.get("h6", {})
    if "baseline" in h6:
        b = h6["baseline"]
        w(f"- **Descriptive (strong):** {(_json.get('discovery') or {}).get('model_name','model')} "
          f"scores {b['acc']:.3f} on MedQA but is badly miscalibrated "
          f"(ECE {b['ece']:.2f}, {b['confidently_wrong_pct']:.0f}% answered >0.9-confident "
          f"and wrong).")
    if _json.get("h4", {}).get("trap_n"):
        h4 = _json["h4"]
        w(f"- **Descriptive (strong):** zero refusal — committed on "
          f"{h4['trap_commit']}/{h4['trap_n']} trap prompts.")
    if h6 and "baseline" in h6 and all(
            abs(h6[c]["acc"] - h6["baseline"]["acc"]) < 0.02
            for c in h6 if c not in ("baseline", "h3_zero_layer")):
        w("- **Causal (null):** neuron interventions do not move the benchmark — "
          "discovery correlations don't transfer to causal control.")
    if _json.get("h3", {}).get("saturated"):
        w("- **Broken:** H3 patching is saturated (score=1.0 everywhere) — not usable.")
    if "h7" not in _json:
        w("- **Next:** run phase 03 (H7) — the FDR scale-up that confirms or refutes H5.")

    report_md = OUT / "report.md"
    report_md.write_text("\n".join(_md) + "\n")
    (OUT / "report.json").write_text(json.dumps(_json, indent=2))
    print("\n" + "=" * 70)
    print("Wrote", report_md)
    print("Wrote", OUT / "report.json")


if __name__ == "__main__":
    main()
