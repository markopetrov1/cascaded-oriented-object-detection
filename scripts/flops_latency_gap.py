#!/usr/bin/env python3
"""The gate is cheap in FLOPs and expensive in wall-clock.

The savings identity has a gate-overhead term g = G_gate / G_detector. Which
currency g is measured in turns out to matter enormously. On the Quadro RTX 8000
(reports/speed/*_b*.json):

    g in GFLOPs                   0.33 %
    g in ms/img at batch 1       27.64 %      83x larger
    g in ms/img at batch 128/16   1.55 %       5x larger

A MobileNetV3-large gate is 0.33 % of the detector's arithmetic but, run one
tile at a time, 28 % of its latency: 5.96 ms against 21.55 ms. The gate is far
too small to saturate the GPU, so its runtime is kernel-launch and Python
overhead rather than compute, while the detector is already compute-bound and
barely improves with batching at all (21.55 -> 20.49 ms from batch 1 to 16).

Substituting the latency g back into the identity changes the conclusions:

  * savings shrink by roughly 27 pp everywhere at batch 1;
  * HRSC2016 goes negative -- the cascade costs more time than it saves;
  * gating all sixteen classes at once collapses from 30 % to 3 %;
  * the three classes whose gates cannot filter at all pay a flat 28 % penalty.

Two honest caveats belong with this result. The batch-1 figure reflects a naive
PyTorch deployment; a fused, traced or TensorRT gate would close much of the
gap, so 27.64 % is a property of this implementation rather than of cascades.
And it is precisely why OAN attaches its objectness head to the detector's own
backbone: a fused head has no second forward pass to launch, which is a real
architectural advantage this paper's decoupled design gives up.

Writes reports/figures/flops_latency_gap.json and the matching figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from figstyle import INK, INK_MUTED, MARKERS, PALETTE, TEXT_WIDTH, save, use_paper_style

import matplotlib.pyplot as plt

SPEED = Path("reports/speed")
SUMMARY = Path("reports/figures/savings_summary.json")


def cost_ratios() -> dict:
    """g in each currency, from the measured batch-size sweep."""
    def load(name):
        return json.load(open(SPEED / name))

    det = {b: load(f"yolo11m_multiclass_obb_b{b}.json") for b in (1, 4, 8, 16)}
    gate = {b: load(f"gate_mbv3large_b{b}.json") for b in (1, 32, 128)}
    g_flops = gate[1]["gflops_per_image"] / det[1]["gflops_per_image"]
    return {
        "gate_gflops": gate[1]["gflops_per_image"],
        "detector_gflops": det[1]["gflops_per_image"],
        "g_flops": g_flops,
        "detector_ms": {b: d["ms_per_image_gpu"] for b, d in det.items()},
        "gate_ms": {b: d["ms_per_image_gpu"] for b, d in gate.items()},
        "g_latency": {
            "batch 1": gate[1]["ms_per_image_gpu"] / det[1]["ms_per_image_gpu"],
            "batch 32/8": gate[32]["ms_per_image_gpu"] / det[8]["ms_per_image_gpu"],
            "batch 128/16": gate[128]["ms_per_image_gpu"] / det[16]["ms_per_image_gpu"],
        },
    }


def main() -> int:
    if not SUMMARY.exists():
        print(f"  missing {SUMMARY} -- run scripts/savings_model.py first")
        return 1
    r = cost_ratios()
    summary = json.load(open(SUMMARY))
    g_flops = r["g_flops"]

    # The fused-gate arms share the detector's forward pass, so their overhead
    # term is zero in every currency. Re-pricing them with the decoupled gate's
    # latency ratio would produce a number that contradicts \S fusing, so they
    # are excluded here and reported there instead.
    FUSED_OR_MATCHED = {"OAN-joint/ships", "Independent/ships-matched"}
    domains = []
    for d in summary:
        best = d["by_tolerance_rule"]["absolute"]
        if not best or d["domain"] in FUSED_OR_MATCHED:
            continue
        # Recover the accept rate from the FLOPs-currency saving, then re-price it.
        accept = 1.0 - best["saved_detector_only"] - g_flops
        entry = {
            "domain": d["domain"], "p_plus": d["p_plus"], "accept_rate": accept,
            "saved_flops": 1.0 - g_flops - accept,
        }
        for k, g in r["g_latency"].items():
            entry[f"saved_latency[{k}]"] = 1.0 - g - accept
        domains.append(entry)

    print("gate/detector cost ratio g")
    print(f"  GFLOPs              {g_flops:.5f}  ({100*g_flops:.2f} %)")
    for k, g in r["g_latency"].items():
        print(f"  ms/img, {k:<13}{g:.5f}  ({100*g:.2f} %)  {g/g_flops:.0f}x the FLOPs ratio")
    print()
    print(f"{'domain':<24}{'p+':>7}{'FLOPs':>9}{'lat b1':>9}{'lat b128':>10}")
    for e in domains:
        print(f"{e['domain']:<24}{e['p_plus']:>7.3f}{100*e['saved_flops']:>8.1f}%"
              f"{100*e['saved_latency[batch 1]']:>8.1f}%{100*e['saved_latency[batch 128/16]']:>9.1f}%")

    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    (out / "flops_latency_gap.json").write_text(json.dumps(
        {"cost_ratios": r, "domains": domains}, indent=2))

    # ---- figure ----------------------------------------------------------
    use_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) why: latency per image against batch size, for both stages
    bd = sorted(r["detector_ms"]); bg = sorted(r["gate_ms"])
    ax1.plot(bd, [r["detector_ms"][b] for b in bd], marker=MARKERS[0],
             color=PALETTE[0], label="detector (184 GFLOPs)")
    ax1.plot(bg, [r["gate_ms"][b] for b in bg], marker=MARKERS[1],
             color=PALETTE[1], label="gate (0.61 GFLOPs)")
    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xlabel("Batch size"); ax1.set_ylabel("ms per image")
    ax1.set_title("(a) The gate never saturates the GPU", loc="left")
    ax1.legend(fontsize=6)
    ax1.annotate("300x fewer FLOPs,\nonly 3.6x faster", xy=(1.05, 5.7),
                 xytext=(3.0, 2.6), fontsize=5.8, color=INK_MUTED, ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=INK_MUTED,
                                 connectionstyle="arc3,rad=0.25"))

    # (b) so what: savings re-priced in latency
    keep = [e for e in domains if e["domain"] != "DOTA-ships/YOLO11n"]
    keep.sort(key=lambda e: e["p_plus"])
    y = np.arange(len(keep)); h = 0.38
    ax2.barh(y - h/2, [100*e["saved_flops"] for e in keep], height=h,
             color=PALETTE[0], label="GFLOPs", zorder=3)
    ax2.barh(y + h/2, [100*e["saved_latency[batch 1]"] for e in keep], height=h,
             color=PALETTE[3], label="ms/img, batch 1", zorder=3)
    ax2.axvline(0, color=INK, lw=0.8, zorder=4)
    labels = [e["domain"].replace("sweep/", "").replace("DOTA-", "") for e in keep]
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=5.4)
    ax2.invert_yaxis()
    ax2.set_xlabel("Compute saved (%)")
    ax2.set_title("(b) Re-priced in wall-clock, savings shrink ~27 pp", loc="left")
    ax2.legend(fontsize=6, loc="lower right")
    ax2.grid(axis="y", visible=False)
    neg = [e for e in keep if e["saved_latency[batch 1]"] < 0]
    if neg:
        ax2.annotate(f"{len(neg)} go negative: the gate\ncosts more than it saves",
                     xy=(-27, 7.5), fontsize=5.4, color=INK_MUTED,
                     ha="left", va="center")
    save(fig, "flops_latency_gap")
    print(f"\n  wrote {out}/flops_latency_gap.json and figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
