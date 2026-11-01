#!/usr/bin/env python3
"""Figures for the revised manuscript, from reports/r1/ only.

    sparsity_ceiling_r1   saving against p+ (DOTA sweep, HRSC2016, SODA-A) and
                          gate recall against tiles removed, with OAN overlaid
    master_pareto_r1      AP against arithmetic saved, per gate backbone
    savings_provenance_r1 the detector budget split four ways per task
    flops_latency_r1      latency against batch size, and operating points
                          priced in arithmetic and in time
    tap_study_r1          fused gates by backbone tap (when the study exists)

Written to reports/r1/figures/; copy deliberately into paper/figures/.

    python scripts/r1_paper_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from figstyle import (COL_WIDTH, DASHES, HATCHES, INK, INK_MUTED, MARKERS, PALETTE, TEXT_WIDTH,  # noqa: E402
                      save, use_paper_style)

import matplotlib.pyplot as plt  # noqa: E402

R1 = REPO / "reports/r1"
OUT = str(R1 / "figures")
BB_NAMES = {"mbv3small": "MobileNetV3-S", "mbv3large": "MobileNetV3-L", "resnet18": "ResNet-18",
            "resnet50": "ResNet-50", "effb0": "EfficientNet-B0", "tiny": "tiny"}


def load(rel):
    p = R1 / rel
    return json.load(open(p)) if p.exists() else None


def sweep_runs():
    return sorted((json.load(open(p)) for p in (R1 / "cascade/sweep").glob("*.json")),
                  key=lambda d: d["risk"]["pooled"]["p_plus"])


def figure_ceiling() -> None:
    runs = sweep_runs()
    hrsc = load("cascade/hrsc/ship.json")
    soda = [load(f"cascade/soda_sweep/{t}.json") for t in ("ship", "plane", "small_vehicle")]
    prior = json.load(open(REPO / "reports/figures/prior_work_prediction.json"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9))
    xs = np.linspace(0, 1, 200)
    ax1.plot(100 * xs, 100 * (1 - xs), "--", color=INK, lw=1.2, zorder=4, label=r"lossless ceiling $1-p_+$")
    ax1.fill_between(100 * xs, 100 * (1 - xs), 100, color="0.955", zorder=0)
    ax1.scatter([100 * d["risk"]["pooled"]["p_plus"] for d in runs], [100 * d["risk"]["pooled"]["saving"] for d in runs],
                s=40, color=PALETTE[0], marker=MARKERS[0], edgecolor="white", lw=0.7, zorder=6,
                label="DOTA-1.5, held-out selection")
    if hrsc:
        t = hrsc["transfer"]
        ax1.scatter([100 * t["full_pass"]["p_plus"]], [100 * t["risk"]["saving"]], s=46, color=PALETTE[3],
                    marker=MARKERS[3], edgecolor="white", lw=0.7, zorder=6, label="HRSC2016, DOTA-chosen point")
    sd = [d for d in soda if d]
    if sd:
        ax1.scatter([100 * d["transfer"]["full_pass"]["p_plus"] for d in sd],
                    [100 * d["transfer"]["risk"]["saving"] for d in sd], s=44, color=PALETTE[2], marker=MARKERS[2],
                    edgecolor="white", lw=0.7, zorder=6, label="SODA-A, DOTA-chosen point")
    anyc = [d for d in runs if d["task"] == "any"]
    if anyc:
        d = anyc[0]["risk"]["pooled"]
        ax1.annotate("any class", (100 * d["p_plus"], 100 * d["saving"]), xytext=(-26, 18),
                     textcoords="offset points", fontsize=5.8, color=INK_MUTED,
                     arrowprops=dict(arrowstyle="-", lw=0.5, color=INK_MUTED))
    ax1.set_xlabel(r"Positive-tile rate $p_+$ (%)")
    ax1.set_ylabel("Detector arithmetic saved (%)")
    ax1.set_xlim(0, 100); ax1.set_ylim(-4, 100)
    ax1.set_title("(a) Savings against sparsity", loc="left")
    ax1.legend(loc="upper right", fontsize=5.4, labelspacing=0.35, borderpad=0.3, handletextpad=0.4)

    show = [("plane", "plane"), ("ship", "ship"), ("small vehicle", "small vehicle"), ("any", "any class")]
    for i, (task, lab) in enumerate(show):
        d = [r for r in runs if r["task"] == task]
        if not d:
            continue
        d = d[0]
        cur = sorted(d["curves"]["mbv3large"], key=lambda r: 1 - r["accept_rate"])
        ax2.plot([100 * (1 - r["accept_rate"]) for r in cur], [100 * r["tpr"] for r in cur], color=PALETTE[i],
                 lw=1.3, label=lab, zorder=3)
        ax2.axvline(100 * (1 - d["risk"]["pooled"]["p_plus"]), color=PALETTE[i], lw=0.8, ls=":", alpha=0.75)
    t9 = prior["oan"]["table9_removal_curve"]
    ax2.plot([100 * r["removed"] for r in t9], [100 * r["recall"] for r in t9], color=PALETTE[5], marker=MARKERS[5],
             ms=3.5, lw=1.6, ls="--", label="OAN (published)", zorder=5)
    lo, hi = prior["oan"]["predicted_lossless_removal_band"]
    ax2.axvspan(100 * lo, 100 * hi, color=PALETTE[5], alpha=0.10, zorder=0)
    ax2.set_xlabel("Tiles removed by the gate (%)")
    ax2.set_ylabel("Gate recall on positive tiles (%)")
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 103)
    ax2.set_title("(b) Recall falls past the ceiling", loc="left")
    ax2.legend(loc="lower left", fontsize=6)
    save(fig, "sparsity_ceiling_r1", OUT)


def figure_pareto() -> None:
    comp = load("speed/components.json")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), sharey=False)
    for ax, (task, title) in zip(axes, (("ship", "ships"), ("plane", "planes"), ("small_vehicle", "small vehicles"))):
        d = load(f"cascade/pillars/{task}.json")
        if d is None:
            continue
        det_g = d["detector_gflops"]
        full = d["full_pass"]["mAP50_coco"]
        for i, (bb, name) in enumerate(BB_NAMES.items()):
            g = d["gates"][bb]["gflops"] / det_g
            cur = sorted(d["curves"][bb], key=lambda r: r["accept_rate"])
            ax.plot([100 * (1 - r["accept_rate"] - g) for r in cur], [r["mAP50_coco"] for r in cur], color=PALETTE[i],
                    lw=1.0, dashes=DASHES[i] if DASHES[i][0] else (None, None), label=name)
        pp = d["full_pass"]["p_plus"]
        ax.axvline(100 * (1 - pp), color=INK, ls="--", lw=0.9)
        ax.scatter([0], [full], marker="*", s=70, color="#C0392B", zorder=6)
        ax.axhline(full - 0.03, color=INK_MUTED, lw=0.6, ls=":")
        ax.set_xlim(-5, 100)
        ax.set_ylim(max(0, full - 0.25), min(1, full + 0.03))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Detector arithmetic saved (%)")
    axes[0].set_ylabel(r"AP$@0.5$ (image level)")
    axes[0].legend(loc="lower left", fontsize=5.2, ncol=1)
    save(fig, "master_pareto_r1", OUT)


def figure_provenance() -> None:
    items = []
    for rel, lab in (("cascade/pillars/plane.json", "plane"), ("cascade/pillars/ship.json", "ship"),
                     ("cascade/pillars/small_vehicle.json", "small vehicle"), ("cascade/sweep/any.json", "any class")):
        d = load(rel)
        if d:
            r = d["risk"]["pooled"]
            items.append((lab, r["p_plus"], r["tpr"], r["fpr"], r["gate_gflops"] / d["detector_gflops"]))
    for t in ("ship", "plane", "small_vehicle"):
        d = load(f"cascade/soda_sweep/{t}.json")
        if d:
            r = d["transfer"]["risk"]
            items.append((f"SODA-A {t.replace('_', ' ')}", r["p_plus"], r["tpr"], r["fpr"],
                          d["gates"]["mbv3large"]["gflops"] / d["detector_gflops"]))
    h = load("cascade/hrsc/ship.json")
    if h:
        r = h["transfer"]["risk"]
        items.append(("HRSC2016", r["p_plus"], r["tpr"], r["fpr"], h["gates"]["mbv3large"]["gflops"] / h["detector_gflops"]))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 0.42 * len(items) + 1.0))
    y = np.arange(len(items))
    hgt = 0.62
    segs = [("detector runs, tile has objects", PALETTE[0], HATCHES[0]),
            ("detector runs, tile is empty", PALETTE[3], HATCHES[3]),
            ("skipped, tile had objects (recall at risk)", PALETTE[1], HATCHES[1]),
            ("skipped, tile was empty (free saving)", PALETTE[2], HATCHES[2])]
    vals = np.array([[p * tpr, (1 - p) * fpr, p * (1 - tpr), (1 - p) * (1 - fpr)] for _, p, tpr, fpr, _ in items])
    left = np.zeros(len(items))
    for k, (lab, col, hat) in enumerate(segs):
        ax.barh(y, vals[:, k] - 0.004, left=left, height=hgt, color=col, hatch=hat, edgecolor="white", lw=0.6,
                label=lab, zorder=3)
        left += vals[:, k]
    for yi, (drop, skip) in zip(y, vals[:, 2:]):
        tot = drop + skip
        if tot > 0.02:
            share = drop / tot
            ax.annotate(f"{share:.0%}", xy=(1.02, yi), xycoords=ax.get_yaxis_transform(), ha="left", va="center",
                        fontsize=7.5, color=INK if share > 0.5 else INK_MUTED, weight="bold" if share > 0.5 else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}\n$p_+$={p:.2f}" for lab, p, *_ in items])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the full-pass detector budget")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=2, fontsize=6)
    save(fig, "savings_provenance_r1", OUT)


def figure_latency() -> None:
    c = load("speed/components_merged.json") or load("speed/components.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7))
    for i, (key, lab, grp) in enumerate((("mbv3large", "MobileNetV3-L gate", "gates"),
                                         ("yolo11m_ships_p50", "YOLO11m-OBB detector", "detectors"),
                                         ("yolo11n_ships", "YOLO11n-OBB detector", "detectors"))):
        e = c[grp][key]["fp32"]
        bs = sorted(int(b) for b in e)
        ax1.plot(bs, [e[str(b)]["median_ms"] / b for b in bs], marker=MARKERS[i], color=PALETTE[i], label=lab)
    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xlabel("Batch size"); ax1.set_ylabel("ms per tile (FP32)")
    ax1.set_title("(a) Per-tile latency", loc="left")
    ax1.legend(fontsize=6)
    # operating points of the DOTA sweep re-priced: S = 1 - a - g, with g in each currency
    det = c["detectors"]["yolo11m_multiclass"]
    gate = c["gates"]["mbv3large_sweep"]   # the sweep gates: same architecture as mbv3large
    g_flops = gate["gflops"] / det["gflops"]
    g_one = (gate["fp32"]["1"]["median_ms"]) / det["fp32"]["1"]["median_ms"]
    g_bat = (gate["fp32"]["32"]["median_ms"] / 32) / (det["fp32"]["16"]["median_ms"] / 16)
    runs = sweep_runs()
    a = np.array([d["risk"]["pooled"]["accept_rate"] for d in runs])
    x = np.arange(len(runs))
    for i, (g, lab) in enumerate(((g_flops, "arithmetic"), (g_bat, "latency, batched"), (g_one, "latency, batch 1"))):
        ax2.scatter(x, 100 * (1 - a - g), marker=MARKERS[i], color=PALETTE[i], s=18, label=lab, zorder=4)
    ax2.axhline(0, color=INK_MUTED, lw=0.6)
    ax2.set_xticks(x)
    abbr = {"basketball court": "BC", "baseball diamond": "BD", "swimming pool": "SP", "soccer ball field": "SBF",
            "roundabout": "RA", "tennis court": "TC", "ground track field": "GTF", "storage tank": "ST",
            "bridge": "BR", "harbor": "HA", "plane": "PL", "large vehicle": "LV", "ship": "SH",
            "small vehicle": "SV", "any": "any"}
    ax2.set_xticklabels([abbr.get(d["task"], d["task"]) for d in runs], fontsize=5.5, rotation=90)
    ax2.set_ylabel("Saved (%)")
    ax2.set_title("(b) One operating point, three currencies", loc="left")
    ax2.legend(fontsize=6, loc="lower left")
    save(fig, "flops_latency_r1", OUT)


def figure_taps() -> None:
    """Tap-depth study, one column: (a) saving at the held-out operating point against the tap layer,
    with the decoupled gate and the jointly trained OAN head on the same scale; (b) the head's PR-AUC
    and the prefix every tile pays. Seed spread drawn when the seed repeats exist."""
    ks = [2, 4, 6, 8, 10]
    runs = [load(f"cascade/tap/tap{k}.json") for k in ks]
    if not all(runs):
        print("tap study results missing; skipping tap figure")
        return
    seeds = [[load(f"cascade/tap/tap{k}_seed{s}.json") if (R1 / f"cascade/tap/tap{k}_seed{s}.json").exists() else None
              for s in (1, 2)] for k in ks]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_WIDTH, 3.7), sharex=True,
                                   gridspec_kw={"hspace": 0.35})
    sav = np.array([100 * r["risk"]["pooled"]["saving"] for r in runs])
    if all(all(s) for s in seeds):
        allv = np.array([[100 * x["risk"]["pooled"]["saving"] for x in [r] + s] for r, s in zip(runs, seeds)])
        ax1.errorbar(ks, allv.mean(1), yerr=allv.std(1, ddof=1), marker="o", color=PALETTE[0], capsize=2,
                     label="head at layer $k$, frozen trunk (3 seeds)")
    else:
        ax1.plot(ks, sav, marker="o", color=PALETTE[0], label="head at layer $k$, frozen trunk")
    # reference lines: mean over the detector seeds that exist (67, 1, 2), as in the fused-gate table
    ref = lambda kind: [d for d in (load(f"cascade/{kind}/{n}.json") for n in ("ship", "ship_seed1", "ship_seed2")) if d]
    ind, fus = ref("indep"), ref("fused")
    if ind:
        ax1.axhline(100 * np.mean([d["risk"]["pooled"]["saving"] for d in ind]), color=PALETTE[2], ls="--",
                    label="decoupled gate")
    if fus:
        ax1.axhline(100 * np.mean([d["risk"]["pooled"]["saving"] for d in fus]), color=PALETTE[3], ls=":",
                    label="OAN head at layer 10, joint training")
    ax1.set_ylabel("Arithmetic saved (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("(a) Saving at the held-out operating point", loc="left")
    ax1.legend(fontsize=6, loc="lower left")
    pr_of = lambda x: x["selections"]["risk"]["all"]["select_fold_prauc"]
    pr = ([float(np.mean([pr_of(x) for x in [r] + s])) for r, s in zip(runs, seeds)]
          if all(all(s) for s in seeds) else [pr_of(r) for r in runs])
    l1 = ax2.plot(ks, pr, marker="s", color=PALETTE[1], label="head PR-AUC")
    ax2.set_ylabel("PR-AUC")
    ax2b = ax2.twinx()
    l2 = ax2b.plot(ks, [100 * r["fused_cost"]["prefix"] / r["detector_gflops"] for r in runs], marker="^",
                   color=INK_MUTED, ls="--", label="prefix, % of detector")
    ax2b.set_ylabel("Prefix (% of detector)")
    ax2b.set_ylim(0, 100)
    ax2b.grid(False)
    ax2.set_xlabel("Backbone layer $k$ the head reads")
    ax2.set_xticks(ks)
    ax2.set_title("(b) Head quality and the prefix every tile pays", loc="left")
    ax2.legend(l1 + l2, [l.get_label() for l in l1 + l2], fontsize=6, loc="lower right")
    save(fig, "tap_study_r1", OUT)


def main() -> int:
    use_paper_style()
    for fn in (figure_ceiling, figure_pareto, figure_provenance, figure_latency, figure_taps):
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
