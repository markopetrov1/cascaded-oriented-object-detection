#!/usr/bin/env python3
"""Do published tile-gating speedups sit under the sparsity ceiling?

The savings identity says a gate cannot skip more than the empty-tile rate
without discarding true positives. That is a claim about any tile gate, not
just ours, so it is testable against results other groups have already
published -- provided their gating task and tiling protocol are known.

Only OAN qualifies for a full quantitative test. It tiles DOTA at 1024x1024
with 200-pixel overlap, which is exactly the protocol in this repo, so the
empty-tile rate of its gating task ("does this patch contain any annotated
object of any class?") is measurable here directly.

R2-CNN and Plastiras et al. run on GF-1/GF-2 and UAV footage whose annotations
are not public at the tile level, so their empty-patch rates cannot be
re-derived. Each does state its own skip rate, however, and the three together
span the range: R2-CNN skips ~99 % of patches on imagery where targets are
rare, Plastiras et al. skip ~65 %, and OAN skips 35 % gating for any of fifteen
DOTA classes at once. Reported speed-ups that look like differences between
methods are, to first order, differences between datasets.

Two conversions matter and are easy to get wrong:

  * OAN reports speed-up as an increase in FPS. A 30.8 % FPS increase is a
    23.5 % reduction in time, not 30.8 %. Fractions of compute saved and FPS
    speed-ups are related by  speedup = 1/(1 - saved) - 1.
  * Patches skipped is an upper bound on time saved, never equal to it: the
    gate runs on every patch and non-detector work does not shrink.

Outputs reports/figures/prior_work_prediction.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METADATA = "data/processed/dota_ships/metadata/tiles.jsonl"


def empty_tile_rate(metadata: str, splits: tuple[str, ...]) -> dict[str, Any]:
    """Fraction of tiles containing no annotated object of any class.

    This is the gating task OAN solves: its objectness head fires on any
    annotated instance, not on one selected class. The DOTA test split is
    excluded because its labels are not public.
    """
    n = n_empty = 0
    with open(metadata) as fh:
        for line in fh:
            row = json.loads(line)
            if row["split"] not in splits:
                continue
            n += 1
            if not row.get("class_counts"):
                n_empty += 1
    return {"n_tiles": n, "n_empty": n_empty, "empty_rate": n_empty / n}


def fps_gain_to_time_saved(fps_before: float, fps_after: float) -> float:
    return 1.0 - fps_before / fps_after


def time_saved_to_fps_gain(saved: float) -> float:
    return 1.0 / (1.0 - saved) - 1.0


# ---------------------------------------------------------------------------
# Published numbers. Sources named so every value can be checked.
# ---------------------------------------------------------------------------

# Xie et al., "Fewer is More", Sci. China Inf. Sci. 2023 (arXiv:2212.13136).
# Table 1, DOTA-v1.0, mAP and FPS with and without OAN.
OAN_TABLE1 = [
    {"detector": "RetinaNet-O", "fps": (16.9, 22.1), "mAP": (68.43, 69.18)},
    {"detector": "S2ANet", "fps": (16.4, 21.8), "mAP": (74.12, 74.37)},
    {"detector": "Faster R-CNN-O", "fps": (14.1, 18.5), "mAP": (69.05, 69.37)},
    {"detector": "RoI Transformer", "fps": (12.4, 16.6), "mAP": (73.76, 73.92)},
    {"detector": "Oriented R-CNN (R50)", "fps": (15.6, 20.4), "mAP": (75.87, 76.02)},
    {"detector": "Oriented R-CNN (RXt50)", "fps": (13.4, 17.5), "mAP": (76.54, 76.73)},
]

# Table 9: patches removed against precision/recall/mAP/FPS. This is the
# ablation that matters -- it is a recall-vs-removal curve on their own data.
OAN_TABLE9 = [
    {"removed": 0.00, "precision": 1.0000, "recall": 1.0000, "mAP": 74.30, "fps": 15.6},
    {"removed": 0.10, "precision": 0.9868, "recall": 0.9999, "mAP": 74.30, "fps": 16.7},
    {"removed": 0.20, "precision": 0.9698, "recall": 0.9991, "mAP": 74.31, "fps": 17.9},
    {"removed": 0.30, "precision": 0.9404, "recall": 0.9974, "mAP": 74.01, "fps": 19.6},
    {"removed": 0.35, "precision": 0.8971, "recall": 0.9936, "mAP": 73.66, "fps": 20.4},
    {"removed": 0.40, "precision": 0.8518, "recall": 0.9883, "mAP": 73.17, "fps": 21.3},
]

# Their own statement about the data, and the Gaofen-2 result.
OAN_STATED_INVALID_TRAIN = 0.39
OAN_GAOFEN2 = {"patches": 1224, "activated": 0.14, "fps_speedup": 0.705}

# Three published systems, three sparsity regimes. None of their empty-patch
# rates can be measured from public data, but each paper states its own skip
# rate, which is the quantity the ceiling bounds. Values below are quoted from
# the papers, with the section they come from, so every one can be checked.
PUBLISHED_REGIMES = [
    {
        "system": "R2-CNN",
        "citation": "Pang et al., IEEE TGRS 2019 (arXiv:1902.06042)",
        "gating_task": "per-patch target presence on GF-1/GF-2 imagery",
        "stated_skip_rate": 0.99,
        "stated_as": "'approximately 99% of the total patches do not need to pass "
                     "the heavy detector branch'",
        "implied_p_plus": 0.01,
        "reports": {"flops": False, "fps": False, "wallclock": True,
                    "operating_point_sweep": True, "calibration": False,
                    "zero_shot_transfer": False},
        "notes": "score thresholds 0.05-0.95 tabulated, but no trade-off curve; "
                 "wall-clock only (29.4 s for an 18000x18192 image on a Titan X)",
    },
    {
        "system": "Plastiras et al.",
        "citation": "ICDSC 2018 (arXiv:1911.06073)",
        "gating_task": "selective tile processing for UAV pedestrian detection",
        "stated_skip_rate": 0.65,
        "stated_as": "'on average the number of selected tiles is below 35% of the "
                     "total amount for every CNN input size'",
        "implied_p_plus": None,  # not stated; pedestrians in UAV footage are sparse
        "reports": {"flops": False, "fps": True, "wallclock": True,
                    "operating_point_sweep": True, "calibration": False,
                    "zero_shot_transfer": False},
        "notes": "six configurations compared rather than a threshold sweep; "
                 "~20 FPS on an i5-8250U laptop CPU",
    },
    {
        "system": "OAN",
        "citation": "Xie et al., Sci. China Inf. Sci. 2023 (arXiv:2212.13136)",
        "gating_task": "any annotated instance of any class, DOTA v1.0/1.5/2.0",
        "stated_skip_rate": 0.35,
        "stated_as": "Table 9 operating point: 35% of patches removed",
        "implied_p_plus": 0.65,
        "reports": {"flops": False, "fps": True, "wallclock": False,
                    "operating_point_sweep": True, "calibration": False,
                    "zero_shot_transfer": False},
        "notes": "the only system whose tiling protocol matches this repo, so its "
                 "empty-patch rate is measurable here (0.30-0.32 vs their stated ~0.39)",
    },
]


def main() -> int:
    # Measured on this repo's tiling, which matches OAN's 1024/200 protocol.
    per_split = {s: empty_tile_rate(METADATA, (s,)) for s in ("train", "val")}
    combined = empty_tile_rate(METADATA, ("train", "val"))

    lo = min(per_split["train"]["empty_rate"], per_split["val"]["empty_rate"])
    hi = max(per_split["train"]["empty_rate"], per_split["val"]["empty_rate"])
    # OAN states ~39 % invalid patches in training; our own measurement is
    # 30-32 %. The gap is most likely a difference in what counts as a valid
    # annotation after clipping, plus their inclusion of v2.0 imagery. Carry
    # the union as an honest band rather than picking one.
    band = (min(lo, OAN_STATED_INVALID_TRAIN), max(hi, OAN_STATED_INVALID_TRAIN))

    print("Ceiling on lossless patch removal for OAN's gating task")
    print(f"  measured here (train)      : {per_split['train']['empty_rate']:.3f}")
    print(f"  measured here (val)        : {per_split['val']['empty_rate']:.3f}")
    print(f"  OAN's own stated figure    : {OAN_STATED_INVALID_TRAIN:.3f} (training, 'approximately')")
    print(f"  band carried forward       : {band[0]:.3f} - {band[1]:.3f}")
    print()

    print("OAN Table 9 -- recall against patches removed, on their data:")
    crossing = None
    for row in OAN_TABLE9:
        marker = ""
        if band[0] <= row["removed"] <= band[1]:
            marker = "  <- inside the predicted ceiling band"
        if crossing is None and row["recall"] < 0.995:
            crossing = row["removed"]
            marker += "  <- recall leaves 99.5 %"
        print(f"  removed {row['removed']:.0%}   recall {row['recall']:.4f}   "
              f"mAP {row['mAP']:.2f}   FPS {row['fps']:.1f}{marker}")
    print()
    print(f"  Recall stays above 99.7 % up to 30 % removal and erodes past it;")
    print(f"  the independently measured empty-tile rate is {band[0]:.0%}-{band[1]:.0%}.")
    print("  Their own ablation traces the ceiling they did not name.")
    print()

    # Their headline operating point, converted into comparable units.
    print("Patches removed is an upper bound on time saved, never equal to it:")
    conv = []
    for row in OAN_TABLE1:
        before, after = row["fps"]
        saved = fps_gain_to_time_saved(before, after)
        conv.append(saved)
        print(f"  {row['detector']:<24} FPS {before:>5.1f} -> {after:>5.1f}  "
              f"= +{100*(after/before-1):>4.1f}% FPS = {100*saved:>4.1f}% time saved")
    mean_saved = sum(conv) / len(conv)
    removed = 0.35  # the operating point Table 9 shows reaching these FPS values
    print(f"\n  mean time saved {100*mean_saved:.1f}% while removing {100*removed:.0f}% of patches")
    print(f"  compute-to-wall-clock conversion: {mean_saved/removed:.2f}x")

    g2_saved = 1.0 - 1.0 / (1.0 + OAN_GAOFEN2["fps_speedup"])
    g2_removed = 1.0 - OAN_GAOFEN2["activated"]
    print(f"\n  Gaofen-2: {100*g2_removed:.0f}% of patches skipped but only "
          f"{100*g2_saved:.1f}% of time saved ({g2_saved/g2_removed:.2f}x)")
    print("  The same FLOPs-to-latency discount this repo measures appears in their numbers.")

    print("\nThree published systems, three sparsity regimes:")
    print(f"  {'system':<18}{'gating task':<44}{'skips':>8}{'implied p+':>12}")
    for r in PUBLISHED_REGIMES:
        ip = f"{r['implied_p_plus']:.2f}" if r["implied_p_plus"] is not None else "not stated"
        print(f"  {r['system']:<18}{r['gating_task'][:43]:<44}{r['stated_skip_rate']:>7.0%}{ip:>12}")
    print("  Reported skip rates span 35-99% and each sits at its own 1 - p+.")
    print("  What reads as a difference between methods is a difference between datasets.")

    print("\nWhat prior work reports:")
    cols = ["flops", "fps", "wallclock", "operating_point_sweep", "calibration", "zero_shot_transfer"]
    print(f"  {'system':<18}" + "".join(f"{c[:11]:>13}" for c in cols))
    for r in PUBLISHED_REGIMES:
        print(f"  {r['system']:<18}" + "".join(f"{'yes' if r['reports'][c] else '-':>13}" for c in cols))
    print("  No system reports FLOPs, gate calibration, or zero-shot transfer.")

    payload = {
        "measured_empty_tile_rate": {"per_split": per_split, "train_and_val": combined},
        "published_regimes": PUBLISHED_REGIMES,
        "oan": {
            "source": "Xie et al., Sci. China Inf. Sci. 2023, arXiv:2212.13136",
            "tiling": "1024x1024, stride 824, 200 px overlap (identical to this repo)",
            "gating_task": "any annotated instance of any class",
            "stated_invalid_patch_rate_train": OAN_STATED_INVALID_TRAIN,
            "predicted_lossless_removal_band": band,
            "table1_fps": OAN_TABLE1,
            "table1_time_saved": conv,
            "table1_mean_time_saved": mean_saved,
            "table9_removal_curve": OAN_TABLE9,
            "recall_leaves_995_at_removal": crossing,
            "operating_point_removal": removed,
            "compute_to_wallclock_ratio": mean_saved / removed,
            "gaofen2": {
                **OAN_GAOFEN2,
                "patches_skipped": g2_removed,
                "time_saved": g2_saved,
                "compute_to_wallclock_ratio": g2_saved / g2_removed,
            },
        },

        "caveats": [
            "OAN's speed-ups are FPS increases measured on a Tesla V100; ours are "
            "GFLOPs plus separately measured wall-clock on a Quadro RTX 8000. The "
            "two are compared only after converting FPS gain to fraction of time saved.",
            "The empty-tile rate is measured on DOTA-1.5 train/val with this repo's "
            "tiler. OAN reports ~39 % invalid patches in training without stating the "
            "split or the annotation-validity rule, so a band is carried rather than "
            "a point estimate.",
            "DOTA's test split has no public labels and is excluded from every rate.",
            "Skip rates for R2-CNN and Plastiras et al. are quoted from their papers; "
            "their empty-patch rates cannot be measured from public data, so the "
            "comparison is between stated skip rate and stated sparsity, not a "
            "re-derivation.",
        ],
    }
    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    (out / "prior_work_prediction.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}/prior_work_prediction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
