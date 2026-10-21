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
def components():
    """Latency from the merged benchmark runs when they exist (scripts/r1_merge_benchmarks.py)."""
    return load("speed/components_merged.json") if (R1 / "speed/components_merged.json").exists() \
        else load("speed/components.json")


def costs() -> None:
    c = components()
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
    num("flopsfactor", det["gflops"] / gate["gflops"], 0)
    num("gatelatratioone", per(det, 1) / per(gate, 1), 1)
    num("gatelatratiobatched", per(det, 16) / per(gate, 32), 0)
    num("gatebatchspeedup", per(gate, 1) / per(gate, 32), 0)
    put("benchwarmup", c["protocol"]["warmup"])
    if "merge" in c:
        pc("benchSpreadMedian", c["merge"]["median_relative_difference"], 1)
        num("benchSpreadMaxFactor", 1 + c["merge"]["max_relative_difference"], 1)
        pc("benchSpreadMax", c["merge"]["max_relative_difference"], 0)
    put("benchiters", c["protocol"]["iters"])


def runtime() -> None:
    r = load("speed/runtime.json")
    if r is None:
        return
    # What the identity predicts from forward-pass costs alone, against what the pipeline
    # measures end to end (the identity ignores preprocessing, decoding and merging).
    c = components()
    if c is not None:
        per = lambda e, b: e["fp32"][str(b)]["median_ms"] / b
        g, d = c["gates"]["mbv3large"], c["detectors"]["yolo11m_ships_p50"]
        p10 = c["prefix"]["10"]["fp32"]
        for mode, M, bg, bd in (("streaming", "Stream", 1, 1), ("batched", "Batch", 32, 16)):
            det = per(d, bd)
            a = r["runs"][f"{mode}/decoupled"]["accept_rate"]
            pc(f"idPred{M}Dec", 1 - a - per(g, bg) / det, 0)
            a = r["runs"][f"{mode}/fused"]["accept_rate"]
            pre = p10[str(bd)]["prefix"]["median_ms"] / bd
            head = p10[str(bd)]["head"]["median_ms"] / bd
            pc(f"idPred{M}Fused", (1 - a) * (1 - pre / det) - head / det, 0)
            st = r["runs"][f"{mode}/full"].get("stage_ms", {})
            tot = r["runs"][f"{mode}/full"].get("profiled_total_ms")
            if tot and "detector" in st:
                pc(f"nonDet{M}", 1 - st["detector"] / tot, 0)
                pc(f"nms{M}", st.get("nms", 0) / tot, 0)
                if mode == "batched":
                    sd = r["runs"]["batched/decoupled"]
                    num("rtBatchGatePerTile", sd["stage_ms"]["gate"] / sd["n_tiles"], 2)
                    num("rtBatchDetPerAcc", sd["stage_ms"]["detector"] / max(1, sd["n_accepted"]), 1)
                    num("rtBatchPrePerTile", sd["stage_ms"]["transfer+preprocess"] / sd["n_tiles"], 1)
                    num("rtTilesPerScene", r["tiles"] / r["scenes"], 0)
                    num("rtAccPerScene", sd["n_accepted"] / r["scenes"], 1)
            else:                       # stage profile comes from the synchronised pass (job 0961)
                MISSING.append("speed/runtime.json stage profile")
                put(f"nonDet{M}", None)
                put(f"nms{M}", None)
    put("rtScenes", r["scenes"])
    put("rtTiles", r["tiles"])
    if c is not None:
        # arithmetic the same accept rates would save, for comparison with the measured time
        a = r["runs"]["batched/decoupled"]["accept_rate"]
        pc("rtArithDecSaved", 1 - a - c["gates"]["mbv3large"]["gflops"] / c["detectors"]["yolo11m_ships_p50"]["gflops"], 1)
    for mode, M in (("streaming", "Stream"), ("batched", "Batch")):
        for kind, K in (("full", "Full"), ("decoupled", "Dec"), ("fused", "Fused")):
            e = r["runs"][f"{mode}/{kind}"]
            num(f"rt{M}{K}", e["ms_per_tile"], 1)
            if kind != "full":
                pc(f"rt{M}{K}Saved", e["time_saved_vs_full"], 0)
                num(f"rt{M}{K}Accept", e["accept_rate"], 3)


def strata_scenes() -> None:
    """Leave-one-scene-out checks of the large stratum losses (scripts/r1_strata_scenes.py)."""
    d = load("stratified/scene_effects.json")
    for key, M in (("ship / source GF", "ShipsGF"), ("ship / source JL", "ShipsJL"),
                   ("plane / gsd 0.3-0.6", "PlanesGsdMid")):
        e = d.get(key) if d else None
        b = e["largest_single_scene"] if e else None
        abs_pp(f"scn{M}Loss", e["dap"] if e else None)
        put(f"scn{M}Scenes", e["scenes"] if e else None)
        abs_pp(f"scn{M}Without", b["dap_without"] if b else None)
        put(f"scn{M}Objects", b["objects"] if b else None)
        put(f"scn{M}Tiles", b["tiles"] if b else None)
        put(f"scn{M}Accepted", b["accepted"] if b else None)


def charged(r: dict) -> float:
    """Share of the skipped detector work that came from target-bearing tiles:
    p+(1-TPR) / [p+(1-TPR) + (1-p+)(1-FPR)]. The gate's own cost is a separate term."""
    drop, skip = r["p_plus"] * (1 - r["tpr"]), (1 - r["p_plus"]) * (1 - r["fpr"])
    return drop / (drop + skip) if drop + skip > 0 else 0.0


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
    pc(f"charged{prefix}", charged(r), 0)
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
    lo = d["results"]["risk_pooled"]["conf_0.001"]["non_difficult"]["overall"]
    num(f"recFullLow{prefix}", lo["recall_full"])
    num(f"recCasLow{prefix}", lo["recall_cascade"])
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
        g = d["by_gsd"].get("0.6-1.5m")
        if g:
            abs_pp(f"strat{T}GsdLow", g["delta_pp"] / 100)
            num(f"strat{T}GsdLowFull", g["mAP50_full"])
        tl = d["tiles_by_largest_gated_object"]
        num(f"strat{T}TprSmallest", tl["0-16px"]["tpr"], 2)
        num(f"strat{T}TprLargest", tl[">=128px"]["tpr"], 2)
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
    c = components()
    if c is not None:   # the same operating point priced in batch-one latency (identity, forward costs)
        g1 = c["gates"]["mbv3large"]["fp32"]["1"]["median_ms"] / c["detectors"]["yolo11m_ships_p50"]["fp32"]["1"]["median_ms"]
        pc("latsaveHRSC", 1 - t["risk"]["accept_rate"] - g1, 0)
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
        pc(f"charged{R}HRSC", charged(x), 0)


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


def fused_extra() -> None:
    for rel, P in (("cascade/fused/ship.json", "Fused"), ("cascade/indep/ship.json", "Indep")):
        d = load(rel)
        if d is not None:
            num(f"prauc{P}", d["selections"]["risk"]["all"]["select_prauc"] if "select_prauc" in d["selections"]["risk"]["all"]
                else d["selections"]["risk"]["all"]["select_fold_prauc"], 3)
            num(f"thrRuntime{P}", d["selections"]["risk"]["all"]["threshold"], 3)
    t = load(str(REPO / "reports/oan/gate_oan_ships_val.threshold.json"))
    if t is not None:
        num("oanStatThreshold", t["threshold"], 3)


def taps() -> None:
    """Tap-depth study: OAN-style heads at backbone layers 2-10 of the frozen 50-epoch ships detector,
    shared-backbone cost with each tap's measured prefix. Seed 67; seeds 1 and 2 give the spread."""
    words = {2: "Two", 4: "Four", 6: "Six", 8: "Eight", 10: "Ten"}
    runs = {k: load(f"cascade/tap/tap{k}.json") for k in words}
    for k, w in words.items():
        d = runs[k]
        x = d["risk"]["pooled"] if d else None
        pc(f"saveTap{w}", x["saving"] if d else None)
        abs_pp(f"dapTap{w}", x["delta_mAP50_coco"] if d else None)
        num(f"accTap{w}", x["accept_rate"] if d else None, 3)
        num(f"praucTap{w}", d["selections"]["risk"]["all"]["select_fold_prauc"] if d else None, 3)
        seeds = [runs[k]] + [load(f"cascade/tap/tap{k}_seed{s}.json") for s in (1, 2)] if d else []
        if seeds and all(seeds):
            v = np.array([100 * r["risk"]["pooled"]["saving"] for r in seeds])
            put(f"sdTap{w}", f"{v.std(ddof=1):.1f}")
        else:
            put(f"sdTap{w}", None)
    c = components()
    if all(runs.values()) and c is not None:
        best = max(words, key=lambda k: runs[k]["risk"]["pooled"]["saving"])
        put("tapBest", best)
        pc("saveTapBest", runs[best]["risk"]["pooled"]["saving"])
        # Time saving the identity predicts for each tap from its measured prefix and head latency
        # (ships detector, batch 1 and per tile at batch 16), as idPred* does for the cascades.
        det = c["detectors"]["yolo11m_ships_p50"]
        per = lambda b: det["fp32"][str(b)]["median_ms"] / b
        for b, name in ((1, "Stream"), (16, "Batch")):
            pred = []
            for k in words:
                f = c["prefix"][str(k)]["fp32"][str(b)]
                pi, eta = f["prefix"]["median_ms"] / b / per(b), f["head"]["median_ms"] / b / per(b)
                pred.append((1 - pi) * (1 - runs[k]["risk"]["pooled"]["accept_rate"]) - eta)
            pc(f"tapTime{name}Best", max(pred), 0)
    else:
        for k in ("tapBest", "saveTapBest", "tapTimeStreamBest", "tapTimeBatchBest"):
            put(k, None)


def teaser() -> None:
    d = load("teaser_facts.json")
    if d is None:
        return
    put("teaserScene", d["scene"])
    put("teaserShips", d["ships"])
    put("teaserTiles", d["tiles"])
    put("teaserSkipped", d["skipped"])
    put("teaserPosTiles", d["positive_tiles"])
    put("teaserLostPos", d["rejected_positive_tiles"])
    pc("teaserOverallRejected", d["overall_rejected_positive_share"], 0)


def detsize() -> None:
    """YOLO11n arm: full pass, MobileNetV3-large cascade, and the PR-AUC rule's choice (ResNet-50)."""
    c = components()
    for task, T in (("ship", "Ships"), ("plane", "Planes"), ("small_vehicle", "SmallVeh")):
        m = load(f"cascade/pillars/{task}.json")
        if m is not None:
            num(f"totM{T}", (1 - m["risk"]["pooled"]["saving"]) * m["detector_gflops"], 1)
        for s, S in ((1024, "Ten"), (896, "Nine"), (768, "Seven")):
            d = load(f"cascade/yolo11n/{task}_imgsz{s}_mbv3large.json")
            r = load(f"cascade/yolo11n/{task}_imgsz{s}.json")
            if d is not None:
                num(f"apN{T}{S}", d["full_pass"]["mAP50_coco"], 3)
                pc(f"saveN{T}{S}", d["risk"]["pooled"]["saving"])
                abs_pp(f"dapN{T}{S}", d["risk"]["pooled"]["delta_mAP50_coco"])
                num(f"totN{T}{S}", (1 - d["risk"]["pooled"]["saving"]) * d["detector_gflops"], 2)
            else:
                for k in ("apN", "saveN", "dapN", "totN"):
                    put(f"{k}{T}{S}", None)
            if r is not None:
                pc(f"saveNRisk{T}{S}", r["risk"]["pooled"]["saving"])
            else:
                put(f"saveNRisk{T}{S}", None)
        if m is not None and load(f"cascade/yolo11n/{task}_imgsz1024_mbv3large.json") is not None:
            d = load(f"cascade/yolo11n/{task}_imgsz1024_mbv3large.json")
            put(f"apN{T}Loss", f"{100 * (m['full_pass']['mAP50_coco'] - d['full_pass']['mAP50_coco']):.0f}")
        else:
            put(f"apN{T}Loss", None)
    if c is not None:
        g50, dn = c["gates"]["resnet50"]["gflops"], c["detectors"]["yolo11n_ships"]["gflops"]
        num("gateRfiftyshareN", 100 * g50 / dn, 0)
        gl = c["gates"]["mbv3large"]["fp32"]["1"]["median_ms"]
        num("gateNlatshareone", 100 * gl / c["detectors"]["yolo11n_ships"]["fp32"]["1"]["median_ms"], 0)


def codesign() -> None:
    import json as _j
    g = load("codesign_gflops.json")
    for v, V in (("shared", "Shared"), ("early_exit", "Early"), ("resnet18", "IndepR")):
        d = load(f"cascade/codesign/{v}.json")
        if d is None:
            for k in ("prauc", "save", "dap"):
                put(f"{k}Cod{V}", None)
            continue
        num(f"praucCod{V}", d["selections"]["risk"]["all"]["select_fold_prauc"], 3)
        pc(f"saveCod{V}", d["risk"]["pooled"]["saving"])
        abs_pp(f"dapCod{V}", d["risk"]["pooled"]["delta_mAP50_coco"])
    if g is not None:
        num("gflopsCodEarly", g["early_exit"], 2)
        num("gflopsCodShared", g["shared"], 2)
    h = load(str(REPO / "runs/codesign_shared_resnet18/history.json"))
    if h is not None:
        num("praucCodOrigShared", max(r["val_pr_auc"] for r in h), 3)
    h = load(str(REPO / "runs/codesign_early_exit_resnet18/history.json"))
    if h is not None:
        num("praucCodOrigEarly", max(r["val_pr_auc"] for r in h), 3)


def prauc_file(rel: str):
    """PR-AUC of a gate score file against its tile labels (all validation tiles)."""
    from sklearn.metrics import average_precision_score
    path = REPO / rel
    if not path.exists():
        MISSING.append(rel)
        return None
    s, y = [], []
    for line in open(path):
        r = json.loads(line)
        s.append(float(r["prob"])); y.append(int(r["label"]))
    return float(average_precision_score(y, s))


def distill() -> None:
    bbs = ["mbv3small", "mbv3large", "resnet18", "resnet50", "effb0", "tiny"]
    dist = {bb: prauc_file(f"reports/r1/distill_scores/gate_ships_{bb}_distill_r1b_val.jsonl") for bb in bbs}
    binr = {bb: prauc_file(f"reports/gate_scores/gate_{bb}_val.jsonl") for bb in bbs}
    if all(v is not None for v in dist.values()):
        num("praucDistLo", min(dist.values()), 3); num("praucDistHi", max(dist.values()), 3)
        diffs = [dist[b] - binr[b] for b in bbs]
        num("praucDistDiffLo", min(diffs), 3); num("praucDistDiffHi", max(diffs), 3)
        num("praucDistDiffAbs", max(abs(d) for d in diffs), 3)
        put("nDistBetter", sum(1 for d in diffs if d > 0))
    else:
        for k in ("praucDistLo", "praucDistHi", "praucDistDiffLo", "praucDistDiffHi", "praucDistDiffAbs", "nDistBetter"):
            put(k, None)
    num("praucBinLo", min(binr.values()), 3); num("praucBinHi", max(binr.values()), 3)
    num("praucDistMbvl", dist["mbv3large"], 3)
    num("praucDistMse", prauc_file("reports/r1/distill_scores/gate_ships_mbv3large_distill_r1_val.jsonl"), 3)
    num("praucDistHard", prauc_file("reports/r1/distill_scores/gate_ships_mbv3large_distill_hard_r1b_val.jsonl"), 3)
    # each binary six-backbone gate evaluated on its own, as the distilled ones are
    sv = {bb: (load(f"cascade/distill/binary_{bb}.json"), load(f"cascade/distill/{bb}.json")) for bb in bbs}
    if all(a and b for a, b in sv.values()):
        s = {bb: (a["risk"]["pooled"], b["risk"]["pooled"]) for bb, (a, b) in sv.items()}
        diffs5 = [s[bb][1]["saving"] - s[bb][0]["saving"] for bb in bbs if bb != "tiny"]
        num("saveDistDiffAbs", 100 * max(abs(x) for x in diffs5), 1)
        pc("saveBinMbvl", s["mbv3large"][0]["saving"]); abs_pp("dapBinMbvl", s["mbv3large"][0]["delta_mAP50_coco"])
        pc("saveBinTiny", s["tiny"][0]["saving"]); pc("saveDistTiny", s["tiny"][1]["saving"])
        abs_pp("dapBinEffb", s["effb0"][0]["delta_mAP50_coco"]); abs_pp("dapDistEffb", s["effb0"][1]["delta_mAP50_coco"])
        num("praucBinEffb", binr["effb0"], 3); num("praucDistEffb", dist["effb0"], 3)
    else:
        for k in ("saveDistDiffAbs", "saveBinMbvl", "dapBinMbvl", "saveBinTiny", "saveDistTiny",
                  "dapBinEffb", "dapDistEffb", "praucBinEffb", "praucDistEffb"):
            put(k, None)
    tg = load("distill_targets.json")
    num("distTargetPrauc", tg["prauc_vs_labels"] if tg else None, 3)
    num("distTargetMeanPos", tg["mean_positive"] if tg else None, 2)
    num("distTargetMeanEmpty", tg["mean_empty"] if tg else None, 2)
    for rel, M in (("cascade/distill/mbv3large.json", "saveDistMbvl"), ("cascade/distill/mbv3large_hard.json", "saveDistHard"),
                   ("cascade/distill/mbv3large_mse.json", "saveDistMse")):
        d = load(rel)
        pc(M, d["risk"]["pooled"]["saving"] if d else None)
        abs_pp(M.replace("save", "dap"), d["risk"]["pooled"]["delta_mAP50_coco"] if d else None)


def overlap() -> None:
    d = load("soda_dota_overlap.json")
    if d is None:
        return
    put("overlapDotaScenes", commas(d["dota_scenes"]))
    put("overlapMedianBits", f"{d['min_distance_per_soda_scene_median']:.0f}")


def soda() -> None:
    """SODA-A transfer: operating points chosen on DOTA val, applied unchanged."""
    for task, T in (("ship", "Ships"), ("plane", "Planes"), ("small_vehicle", "SmallVeh")):
        d = load(f"cascade/soda_sweep/{task}.json")
        if d is None:
            for k in ("fullpassSoda", "saveSoda", "dapSoda", "tprSoda", "fprSoda", "chargedSoda", "ceilSoda",
                      "savePlainSoda", "dapPlainSoda"):
                put(f"{k}{T}", None)
            continue
        t = d["transfer"]
        num(f"fullpassSoda{T}", t["full_pass"]["mAP50_coco"], 3)
        pc(f"ceilSoda{T}", 1 - t["full_pass"]["p_plus"])
        x = t["risk"]
        pc(f"saveSoda{T}", x["saving"]); abs_pp(f"dapSoda{T}", x["delta_mAP50_coco"])
        num(f"tprSoda{T}", x["tpr"], 2); num(f"fprSoda{T}", x["fpr"], 2); pc(f"chargedSoda{T}", charged(x), 0)
        y = t["plain_full"]
        pc(f"savePlainSoda{T}", y["saving"]); abs_pp(f"dapPlainSoda{T}", y["delta_mAP50_coco"])
        num(f"fprDota{T}", d["risk"]["pooled"]["fpr"], 2)
        q = load(f"cascade/soda_pillars/{task}.json")
        if q is not None:
            pc(f"saveSodaRfifty{T}", q["transfer"]["risk"]["saving"])
            abs_pp(f"dapSodaRfifty{T}", q["transfer"]["risk"]["delta_mAP50_coco"])
        else:
            put(f"saveSodaRfifty{T}", None); put(f"dapSodaRfifty{T}", None)


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


def sweep_summary() -> None:
    tasks = sorted(p for p in (R1 / "cascade/sweep").glob("*.json"))
    if not tasks:
        MISSING.append("cascade/sweep/*.json")
        return
    gap, ho_dap, ss_dap = [], [], []
    for p in tasks:
        d = json.load(open(p))
        r, s = d["risk"]["pooled"], d["same_split"]
        name = "".join(w.capitalize() for w in d["task"].split())
        pc(f"saveSweep{name}", r["saving"])
        abs_pp(f"dapSweep{name}", r["delta_mAP50_coco"])
        num(f"praucSweep{name}", d["selections"]["risk"]["all"]["select_fold_prauc"], 3)
        num(f"fprSweep{name}", r["fpr"], 2)
        gap.append(s["saving"] - r["saving"])
        ho_dap.append(r["delta_mAP50_coco"]); ss_dap.append(s["delta_mAP50_coco"])
    pc("selCostMean", float(np.mean(gap)))
    pc("selCostMin", float(np.min(gap)))
    pc("selCostMax", float(np.max(gap)))
    abs_pp("hoDapMean", float(np.mean(ho_dap)))
    abs_pp("ssDapMean", float(np.mean(ss_dap)))
    put("nOverBudget", sum(1 for x in ho_dap if x < -0.03))


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
    strata(); calibration(); hrsc(); prior(); misc(); facts(); sweep_summary(); fused_extra(); teaser(); detsize(); soda(); overlap(); codesign(); distill(); taps(); strata_scenes()
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
