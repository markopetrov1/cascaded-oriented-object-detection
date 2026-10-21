#!/usr/bin/env python3
"""Write paper/generated_numbers.tex for the revised manuscript from reports/r1/.

Every number the revised text quotes is a macro defined here and computed from a
results file, so the prose, tables and figures cannot disagree. A result that
does not exist yet is written as a visible placeholder (a bold "??"), and the
script lists the missing inputs; scripts/r1_check_numbers.py refuses a build
that still contains one.

    python scripts/r1_paper_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
R1 = REPO / "reports/r1"
OUT = REPO / "paper/generated_numbers.tex"

MACROS: list[tuple[str, str, str]] = []
MISSING: list[str] = []


def put(name: str, value, fmt: str = "{}", note: str = "") -> None:
    MACROS.append((name, fmt.format(value) if value is not None else r"\textbf{??}", note))


def load(rel: str):
    p = R1 / rel if not rel.startswith("/") else Path(rel)
    if not p.exists():
        MISSING.append(rel)
        return None
    return json.load(open(p))


def pct(x, d=1):
    return None if x is None else round(100 * x, d)


def pc(name, x, d=1, note=""):
    """Percentage with d decimals (value in [0,1])."""
    put(name, None if x is None else f"{100 * x:.{d}f}", note=note)


def num(name, x, d=3, note=""):
    put(name, None if x is None else f"{x:.{d}f}", note=note)


def signed_pp(name, x, d=2, note=""):
    """Change in percentage points, as a LaTeX number with an explicit sign (minus as $-$)."""
    if x is None:
        put(name, None)
        return
    v = 100 * x
    s = f"{abs(v):.{d}f}"
    put(name, (r"$-$" if v < 0 else "+") + s if round(v, d) != 0 else s, note=note)


def abs_pp(name, x, d=2, note=""):
    put(name, None if x is None else f"{abs(100 * x):.{d}f}", note=note)


# ---------------------------------------------------------------------------
def costs() -> None:
    c = load("speed/components.json")
    if c is None:
        return
    g, d = c["gates"], c["detectors"]
    per = lambda e, b: e["fp32"][str(b)]["median_ms"] / b
    det, det16, gate = d["yolo11m_ships_p50"], d["yolo11m_multiclass"], g["mbv3large"]
    num("detgflops", det16["gflops"], 1, "16-class YOLO11m-OBB at 1024 px")
    num("detshipsgflops", det["gflops"], 1)
    num("gategflops", gate["gflops"], 2, "MobileNetV3-large at 256 px")
    num("gateRfiftygflops", g["resnet50"]["gflops"], 1)
    num("gateRfiftyshare", 100 * g["resnet50"]["gflops"] / det16["gflops"], 1)
    num("detngflops", d["yolo11n_ships"]["gflops"], 1)
    num("gflopsratio", 100 * gate["gflops"] / det["gflops"], 2)
    num("gatelatone", per(gate, 1), 2)
    num("gatelatbatched", per(gate, 32), 2)
    num("gatelatbatchedmax", per(gate, 128), 2)
    num("detlatone", per(det, 1), 2)
    num("detlatbatched", per(det, 16), 2)
    num("detlatbatchedhalf", per(det, 16) if False else det["fp16"]["16"]["median_ms"] / 16, 2)
    num("detlatoneqlo", det["fp32"]["1"]["q25_ms"], 2)
    num("detlatoneqhi", det["fp32"]["1"]["q75_ms"], 2)
    num("gatelatoneqlo", gate["fp32"]["1"]["q25_ms"], 2)
    num("gatelatoneqhi", gate["fp32"]["1"]["q75_ms"], 2)
    num("latratioone", 100 * per(gate, 1) / per(det, 1), 1)
    num("latratiobatched", 100 * per(gate, 32) / per(det, 16), 1)
    num("detnlatone", per(d["yolo11n_ships"], 1), 2)
    num("detnlatbatched", per(d["yolo11n_ships"], 16), 2)
    pre = c["prefix"]
    p10 = pre["10"]
    num("prefixgflops", p10["prefix_gflops"], 1)
    num("prefixshare", 100 * p10["prefix_gflops"] / det["gflops"], 1)
    num("headgflops", p10["head_gflops"], 2)
    num("prefixlatone", p10["fp32"]["1"]["prefix"]["median_ms"], 2)
    num("prefixlatbatched", p10["fp32"]["16"]["prefix"]["median_ms"] / 16, 2)
    num("prefixlatshareone", 100 * p10["fp32"]["1"]["prefix"]["median_ms"] / per(det, 1), 0)
    num("prefixlatsharebatched", 100 * (p10["fp32"]["16"]["prefix"]["median_ms"] / 16) / per(det, 16), 0)
    for k, w in (("2", "Two"), ("4", "Four"), ("6", "Six"), ("8", "Eight"), ("10", "Ten")):
        num(f"prefixgflops{w}", pre[k]["prefix_gflops"], 1)
        num(f"prefixshare{w}", 100 * pre[k]["prefix_gflops"] / det["gflops"], 0)
    put("benchwarmup", c["protocol"]["warmup"])
    put("benchiters", c["protocol"]["iters"])


def runtime() -> None:
    r = load("speed/runtime.json")
    if r is None:
        return
    put("rtScenes", r["scenes"])
    put("rtTiles", r["tiles"])
    for mode, M in (("streaming", "Stream"), ("batched", "Batch")):
        for kind, K in (("full", "Full"), ("decoupled", "Dec"), ("fused", "Fused")):
            e = r["runs"][f"{mode}/{kind}"]
            num(f"rt{M}{K}", e["ms_per_tile"], 1)
            if kind != "full":
                pc(f"rt{M}{K}Saved", e["time_saved_vs_full"], 0)
                num(f"rt{M}{K}Accept", e["accept_rate"], 3)


def cascade(prefix: str, rel: str) -> dict | None:
    d = load(rel)
    if d is None:
        return None
    r, s = d["risk"]["pooled"], d["same_split"]
    b = r["bootstrap"]
    pc(f"posrate{prefix}", r["p_plus"])
    pc(f"ceil{prefix}", 1 - r["p_plus"])
    pc(f"save{prefix}", r["saving"])
    abs_pp(f"dap{prefix}", r["delta_mAP50_coco"])
    abs_pp(f"dapvoc{prefix}", r["delta_mAP50_voc07"])
    num(f"tpr{prefix}", r["tpr"])
    num(f"fpr{prefix}", r["fpr"])
    num(f"acc{prefix}", r["accept_rate"])
    pc(f"charged{prefix}", r["charged_share"] or 0.0, 0)
    num(f"fullpass{prefix}", d["full_pass"]["mAP50_coco"])
    num(f"fullpassvoc{prefix}", d["full_pass"]["mAP50_voc07"])
    pc(f"saveLo{prefix}", b["saving_ci95"][0]); pc(f"saveHi{prefix}", b["saving_ci95"][1])
    abs_pp(f"dapLo{prefix}", b["delta_mAP50_coco_ci95"][0]); signed_pp(f"dapHi{prefix}", b["delta_mAP50_coco_ci95"][1])
    pc(f"ssSave{prefix}", s["saving"])
    abs_pp(f"ssDap{prefix}", s["delta_mAP50_coco"])
    for rule, R in (("plain_full", "Plain"), ("risk_full", "RiskFull")):
        pc(f"save{R}{prefix}", d[rule]["pooled"]["saving"])
        abs_pp(f"dap{R}{prefix}", d[rule]["pooled"]["delta_mAP50_coco"])
    pc(f"gateFree{prefix}", r["saving_if_gate_free"])
    return d


def recall(prefix: str, rel: str) -> None:
    d = load(rel)
    if d is None:
        return
    o = d["results"]["risk_pooled"]["conf_0.25"]["non_difficult"]["overall"]
    num(f"recFull{prefix}", o["recall_full"])
    num(f"recCas{prefix}", o["recall_cascade"])
    put(f"recDelta{prefix}", f"{abs(o['delta_recall_pp']):.2f}")
    put(f"recShareRej{prefix}", f"{o['share_on_rejected_tiles_pp']:.1f}")
    put(f"recLost{prefix}", o["fully_rejected_lost"])
    put(f"recMissed{prefix}", o["fully_rejected_missed_anyway"])
    put(f"recRecovered{prefix}", o["partly_rejected_recovered"])
    put(f"recPartLost{prefix}", o["partly_rejected_lost"])
    put(f"recN{prefix}", f"{o['n_objects']:,}".replace(",", "{,}"))
    fr = o["fully_rejected_missed_anyway"] / max(1, o["on_fully_rejected_tiles"])
    pc(f"recMissedShare{prefix}", fr, 0)
    small = d["results"]["risk_pooled"]["conf_0.25"]["non_difficult"]["by_size"]["0-16"]
    num(f"recSmallFull{prefix}", small["recall_full"])
    num(f"recSmallCas{prefix}", small["recall_cascade"])


def strata() -> None:
    for task, T in (("ship", "Ships"), ("plane", "Planes")):
        d = load(f"stratified/pillars_{task}.json")
        if d is None:
            continue
        src = d["by_source"]
        for s, S in (("GF", "GF"), ("JL", "JL"), ("GoogleEarth", "GE")):
            if s in src and src[s]["mAP50_full"] == src[s]["mAP50_full"]:
                abs_pp(f"strat{T}{S}", src[s]["delta_pp"] / 100)
                num(f"strat{T}{S}Full", src[s]["mAP50_full"])
                num(f"strat{T}{S}Cas", src[s]["mAP50_cascade"])
                put(f"strat{T}{S}Img", src[s]["n_images"])
        tb = d["tiles_by_boundary"]
        num(f"strat{T}FragTpr", tb["fragments only"]["tpr"], 2)
        num(f"strat{T}WholeTpr", tb["whole objects only"]["tpr"], 2)
        put(f"strat{T}AlwaysClipped", d["objects_by_boundary"]["always clipped"]["n_objects"])


def calibration() -> None:
    d = load("cascade/pillars/ship.json")
    if d is not None:
        e = d["ece_heldout"]
        for c, C in (("identity", "Ident"), ("temperature", "Temp"), ("platt", "Platt"), ("isotonic", "Iso"),
                     ("context_adaptive", "Ctx")):
            vals = [e[g][c]["heldout_mean"] for g in e]
            num(f"ece{C}", float(np.mean(vals)), 3)
            num(f"ece{C}Lo", min(vals), 3)
            num(f"ece{C}Hi", max(vals), 3)
    t = load("calibration/ties_ship.json")
    if t is not None:
        st = t["tie_stats"]
        iso = [st[g]["isotonic"][k] for g in st for k in ("fit_A_report_B", "fit_B_report_A")]
        put("isoDistinctLo", min(x["distinct_calibrated"] for x in iso))
        put("isoDistinctHi", max(x["distinct_calibrated"] for x in iso))
        raw = [x["distinct_raw"] for x in iso]
        put("rawDistinctLo", f"{min(raw):,}".replace(",", "{,}"))
        put("rawDistinctHi", f"{max(raw):,}".replace(",", "{,}"))
        num("isoGapLo", min(x["max_accept_gap_pp"] for x in iso), 1)
        num("isoGapHi", max(x["max_accept_gap_pp"] for x in iso), 1)
        num("rawGapHi", max(x["max_accept_gap_raw_pp"] for x in iso), 2)
        o = t["oracle_best_saving"]
        loss = [100 * (np.mean([o[g]["identity"][f]["saving"] for f in "AB"]) -
                       np.mean([o[g]["isotonic"][f]["saving"] for f in "AB"])) for g in o]
        num("isoLossLo", min(loss), 1)
        num("isoLossHi", max(loss), 1)
        inv = sum(st[g][m][k]["inversions"] for g in st for m in ("temperature", "platt", "isotonic")
                  for k in ("fit_A_report_B", "fit_B_report_A"))
        put("calInversions", inv)


def protocol() -> None:
    f = yaml.safe_load(open(REPO / "configs/splits/r1_val_2fold.yaml"))
    put("nFoldA", len(f["A"]))
    put("nFoldB", len(f["B"]))
    put("nValImages", len(f["A"]) + len(f["B"]))
    put("nQuantiles", 200)
    put("bootRisk", 200)
    put("bootCI", 1000)
    tasks = sorted((REPO / "reports/r1/cascade/sweep").glob("*.json"))
    tasks = [t for t in tasks if not t.name.endswith("_masks.npz")]
    put("nDotaTasks", len(tasks))
    put("nGatingTasks", len(tasks) + 1)
    pp = [json.load(open(t))["risk"]["pooled"]["p_plus"] for t in tasks]
    h = load("cascade/hrsc/ship.json")
    pc("posrateMin", min(pp))
    pc("posrateMax", h["transfer"]["full_pass"]["p_plus"] if h else None)
    within = sum(1 for t in tasks if json.load(open(t))["risk"]["pooled"]["delta_mAP50_coco"] >= -0.03)
    put("nWithinBudget", within)
    positive = sum(1 for t in tasks if json.load(open(t))["risk"]["pooled"]["saving"] > 0)
    put("nPositiveSaving", positive)


def hrsc() -> None:
    d = load("cascade/hrsc/ship.json")
    if d is None:
        return
    t = d["transfer"]
    num("fullpassHRSC", t["full_pass"]["mAP50_coco"])
    pc("posrateHRSC", t["full_pass"]["p_plus"])
    pc("ceilHRSC", 1 - t["full_pass"]["p_plus"])
    put("nHRSC", t["full_pass"]["n_tiles"])
    for rule, R in (("risk", ""), ("plain_full", "Plain")):
        x = t[rule]
        pc(f"save{R}HRSC", x["saving"])
        abs_pp(f"dap{R}HRSC", x["delta_mAP50_coco"])
        num(f"tpr{R}HRSC", x["tpr"])
        put(f"fprk{R}HRSC", x["fpr_counts"][0])
        put(f"fprn{R}HRSC", x["fpr_counts"][1])
        num(f"fprLo{R}HRSC", x["fpr_ci95_clopper_pearson"][0], 2)
        num(f"fprHi{R}HRSC", x["fpr_ci95_clopper_pearson"][1], 2)
        pc(f"charged{R}HRSC", x["charged_share"] or 0.0, 0)


def prior() -> None:
    p = load(str(REPO / "reports/figures/prior_work_prediction.json"))
    if p is not None:
        m = p["measured_empty_tile_rate"]["per_split"]
        pc("oanMeasuredTrain", m["train"]["empty_rate"])
        pc("oanMeasuredVal", m["val"]["empty_rate"])
        o = p["oan"]
        pc("oanStated", o["stated_invalid_patch_rate_train"], 0)
        pc("oanTimeSaved", o["table1_mean_time_saved"])
        pc("oanRemoval", o["operating_point_removal"], 0)
        num("oanConversion", o["compute_to_wallclock_ratio"], 2)
        pc("oanGaofenSkipped", o["gaofen2"]["patches_skipped"], 0)
        pc("oanGaofenTime", o["gaofen2"]["time_saved"])
        num("oanGaofenConversion", o["gaofen2"]["compute_to_wallclock_ratio"], 2)
    d = load("prior_work/dota10_empty_rate.json")
    if d is not None:
        for s, S in (("train", "Train"), ("val", "Val")):
            v = d[s]
            pc(f"dotaTen{S}Hi", max(v["empty_rate_dota10_longside"], v["empty_rate_dota10_sqrtarea"]))


def misc() -> None:
    # truncation effect (WP1), measured on the ships p50 detector at tile level
    t = load("truncation_effect.json")
    if t is not None:
        num("truncBefore", t["ap50_coco_conf025"], 3)
        num("truncAfter", t["ap50_coco_conf0001"], 3)
    s = load("object_sizes.json")
    if s is not None:
        num("svMedianAll", s["small vehicle"]["median_all"], 1)
        num("svMedianNonDiff", s["small vehicle"]["median_nondiff"], 1)
        num("shipMedian", s["ship"]["median_all"], 1)
        pc("svDifficultShare", s["small vehicle"]["difficult_share"], 0)
        pc("difficultShare", s["all"]["difficult_share"], 0)
    sv = load("small_vehicle_scene.json")
    if sv is not None:
        put("svSceneLost", sv["lost_in_scene"])
        put("svTotalLost", sv["lost_total"])
        put("svScenePos", sv["positive_tiles"])
        put("svSceneAcc", sv["accepted_tiles"])


def commas(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def facts() -> None:
    d = load("dataset_facts.json")
    if d is None:
        return
    put("nTrainTiles", commas(d["dota_train_tiles"]))
    put("nValTiles", commas(d["dota_val_tiles"]))
    put("densestTile", commas(d["densest_tile_detections_multiclass"]))
    put("nHRSCneg", d["hrsc_test_negatives"])
    s = d["soda"]
    put("nSodaScenes", s["scenes"])
    put("nSodaTiles", commas(s["tiles"]))
    put("nSodaIgnoreTiles", commas(s["tiles_with_ignore"]))
    pc("posrateSodaShips", s["p_plus_ship"])
    pc("posrateSodaPlanes", s["p_plus_plane"])
    pc("posrateSodaSmallVeh", s["p_plus_small_vehicle"])
    put("nGateCheckpoints", d["gate_checkpoints"])
    num("gateGapMedian", d["gate_prauc_best_minus_last_median"], 3)
    num("gateGapMax", d["gate_prauc_best_minus_last_max"], 3)


def main() -> int:
    costs(); runtime(); protocol()
    cascade("Ships", "cascade/pillars/ship.json")
    cascade("Planes", "cascade/pillars/plane.json")
    cascade("SmallVeh", "cascade/pillars/small_vehicle.json")
    cascade("AnyClass", "cascade/sweep/any.json")
    cascade("ShipsSweep", "cascade/sweep/ship.json")
    cascade("ShipsPfifty", "cascade/shipsp50/ship.json")
    cascade("Indep", "cascade/indep/ship.json")
    cascade("Fused", "cascade/fused/ship.json")
    for p, f in (("Ships", "pillars_ship"), ("Planes", "pillars_plane"), ("SmallVeh", "pillars_small_vehicle"),
                 ("Indep", "indep_ship"), ("Fused", "fused_ship")):
        recall(p, f"object_recall/{f}.json")
    strata(); calibration(); hrsc(); prior(); misc(); facts()
    lines = ["% Generated by scripts/r1_paper_numbers.py from reports/r1/ -- do not edit by hand."]
    seen = set()
    for name, val, note in MACROS:
        if name in seen:
            raise SystemExit(f"duplicate macro {name}")
        seen.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}" + (f"  % {note}" if note else ""))
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(MACROS)} macros)")
    if MISSING:
        print("missing inputs (placeholders written):", *sorted(set(MISSING)), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
