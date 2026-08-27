#!/usr/bin/env python3
"""Closed-form model for cascade compute savings, fitted to nothing.

The cascade's per-tile cost is an accounting identity, not an empirical curve.
With N tiles, a gate costing G_g per tile, a detector costing G_f per tile, and
an accept rate a(tau), the total is

    C(tau) = N * G_g + N * a(tau) * G_f .

The accept rate decomposes through the gate's operating point on a population
with positive-tile rate p+:

    a(tau) = p+ * TPR(tau) + (1 - p+) * FPR(tau) .

Writing g = G_g / G_f for the gate's relative overhead, compute saved against a
detector-only baseline (N * G_f) is therefore

    S(tau) = 1 - g - p+*TPR(tau) - (1 - p+)*FPR(tau)
           = p+*(1 - TPR)        <- saved by dropping positive tiles (costs mAP)
           + (1 - p+)*(1 - FPR)  <- saved by correctly skipping empty tiles
           - g                   <- paid back as gate overhead

and the perfect-gate ceiling (TPR = 1, FPR = 0) is S_max = 1 - p+ - g.

The repo's own reports use a slightly different baseline: the tau = 0 row, which
already pays the gate on every tile. Against that baseline the same identity is

    S_repo(tau) = 1 - (g + a(tau)) / (1 + g),  ceiling (1 - p+) / (1 + g) .

Both are emitted so the published numbers stay reproducible while the cleaner
detector-only form is available for the paper.

This script recovers every term for every operating point already on disk and
checks the identity against the recorded `total_gflops`. If the residuals are
not at floating-point level, the model is wrong and nothing downstream of it
should be trusted.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Iterable

# Domains we can decompose, and where their tile population is defined.
# `classes` selects the gating task out of the per-tile `class_counts`.
DOMAINS: list[dict[str, Any]] = [
    {
        "name": "DOTA-ships",
        "metadata": "data/processed/dota_ships/metadata/tiles.jsonl",
        "split": "val",
        "classes": ["ship"],
        "detector": "YOLO11m-OBB",
        "globs": ["reports/cascade/*.json"],
    },
    {
        "name": "DOTA-planes",
        "metadata": "data/processed/dota_planes/metadata/tiles.jsonl",
        "split": "val",
        "classes": ["plane"],
        "detector": "YOLO11m-OBB",
        "globs": ["reports/planes/cascade/*.json"],
    },
    {
        "name": "DOTA-small_vehicle",
        "metadata": "data/processed/dota_small_vehicle/metadata/tiles.jsonl",
        "split": "val",
        "classes": ["small vehicle"],
        "detector": "YOLO11m-OBB",
        "globs": ["reports/small_vehicle/cascade/*.json"],
    },
    {
        "name": "HRSC2016",
        "metadata": "data/processed/hrsc2016/metadata/tiles.jsonl",
        "split": "test",
        "classes": ["ship"],
        "detector": "YOLO11m-OBB (zero-shot)",
        "globs": ["reports/hrsc/*.json"],
    },
    {
        "name": "DOTA-ships/YOLO11n",
        "metadata": "data/processed/dota_ships/metadata/tiles.jsonl",
        "split": "val",
        "classes": ["ship"],
        "detector": "YOLO11n-OBB",
        "globs": ["reports/cascade_yolo11n/*.json"],
    },
]

# Rows that are not plain gate-vs-detector cascades.
_SKIP_SUBSTRINGS = ("stratif", "codesign", "distill", "smoke")

# The 16-class sparsity sweep. Each task is one gate class against the shared
# multi-class detector, so p+ comes from reports/gate_tasks.json (written by
# make_gate_labels.py) rather than from a per-class metadata file.
SWEEP_TASKS = Path("reports/gate_tasks.json")
SWEEP_CASCADE = Path("reports/sweep/cascade")


def sweep_domains() -> list[dict[str, Any]]:
    """Domain entries for whichever sweep tasks have cascade results on disk.

    Returns an empty list before the sweep runs, so this module stays usable
    either side of it.
    """
    if not SWEEP_TASKS.exists() or not SWEEP_CASCADE.exists():
        return []
    domains = []
    for info in json.load(open(SWEEP_TASKS)):
        task = info["task"]
        slug = task.replace(" ", "_")
        # eval_cascade --all-calibrations writes one file per calibration,
        # <out-stem>_<calibration>.json, not the --out path itself.
        pattern = str(SWEEP_CASCADE / f"{slug}_*.json")
        if not glob.glob(pattern):
            continue
        domains.append({
            "name": f"sweep/{task}",
            "p_plus_override": info["val"]["p_plus"],
            "split": "val",
            "detector": "YOLO11m-OBB (16-class)",
            "globs": [pattern],
        })
    return domains


# The OAN replication arm. Its gate is a head on the detector's own backbone, so
# it shares the detector's forward pass and its overhead term is zero by
# construction; the decomposition still applies unchanged.
OAN_CASCADE = Path("reports/oan")


def oan_domains() -> list[dict[str, Any]]:
    """Domain entry for the jointly-trained OAN arm, if it has been evaluated."""
    if not glob.glob(str(OAN_CASCADE / "cascade_oan_ships_*.json")):
        return []
    out = [{
        "name": "OAN-joint/ships",
        "metadata": "data/processed/dota_ships/metadata/tiles.jsonl",
        "split": "val",
        "classes": ["ship"],
        "detector": "YOLO11m-OBB + fused OAN head",
        "globs": [str(OAN_CASCADE / "cascade_oan_ships_*.json")],
    }]
    # The independent gate against the matched 50-epoch control, so the
    # comparison is not confounded by the detector's training budget: the
    # original ships detector early-stopped at epoch 11-14 under patience 10.
    if glob.glob(str(OAN_CASCADE / "cascade_indep_ships_p50_*.json")):
        out.append({
            "name": "Independent/ships-matched",
            "metadata": "data/processed/dota_ships/metadata/tiles.jsonl",
            "split": "val",
            "classes": ["ship"],
            "detector": "YOLO11m-OBB (50 ep) + MobileNetV3 gate",
            "globs": [str(OAN_CASCADE / "cascade_indep_ships_p50_*.json")],
        })
    return out


def positive_rate(metadata: str | Path, split: str, classes: Iterable[str]) -> tuple[float, int, int]:
    """Fraction of tiles in `split` containing at least one instance of `classes`.

    Read from the tiling metadata rather than inferred from an oracle run, so it
    is defined even for domains with no oracle rows (this is what left HRSC2016
    out of reports/figures/sparsity_ceiling.json).
    """
    wanted = set(classes)
    n = n_pos = 0
    with open(metadata) as fh:
        for line in fh:
            row = json.loads(line)
            if row["split"] != split:
                continue
            n += 1
            if wanted & set(row.get("class_counts", {})):
                n_pos += 1
    if n == 0:
        raise ValueError(f"no tiles for split={split!r} in {metadata}")
    return n_pos / n, n_pos, n


def load_rows(globs: Iterable[str]) -> list[dict[str, Any]]:
    """Deduplicated cascade rows, excluding oracle/codesign/distill variants.

    Mirrors the filtering in scripts/sparsity_ceiling.py so the two agree: the
    same (label, calibration, threshold) triple appears in several files, and
    old stratified-step outputs duplicate the per-calibration JSONs.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for pattern in globs:
        for path in sorted(glob.glob(pattern)):
            if any(s in path for s in _SKIP_SUBSTRINGS):
                continue
            for row in json.load(open(path)):
                label, calib = row.get("label", ""), row.get("calibration", "")
                if calib and label.endswith("_" + calib):
                    continue
                key = (label, calib, round(row["threshold"], 3))
                if key in seen:
                    continue
                seen.add(key)
                row = dict(row)
                row["_source"] = path
                rows.append(row)
    return rows


def decompose(row: dict[str, Any], p_plus: float) -> dict[str, Any] | None:
    """Recover every term of the savings identity for one operating point."""
    n_tiles = row["n_tiles"]
    n_accepted = row["n_accepted"]
    if not n_tiles or not n_accepted:
        return None  # tau = 1.0 rejects everything; G_f is unrecoverable there

    g_gate = row["gate_gflops"] / n_tiles
    g_det = row["detector_gflops"] / n_accepted
    if g_det <= 0:
        return None
    g = g_gate / g_det

    accept = n_accepted / n_tiles
    tpr = row["gate_recall_on_positive_tiles"]
    # a = p+*TPR + (1-p+)*FPR, inverted for the false-positive rate.
    fpr = (accept - p_plus * tpr) / (1.0 - p_plus) if p_plus < 1.0 else float("nan")

    saved_detector_only = 1.0 - g - accept
    saved_repo = 1.0 - (g + accept) / (1.0 + g)

    # Identity check: rebuild the recorded total from the model's terms alone.
    predicted_total = n_tiles * g_gate + n_tiles * accept * g_det
    residual = predicted_total - row["total_gflops"]

    return {
        "label": row.get("label"),
        "calibration": row.get("calibration"),
        "gate": row.get("gate"),
        "threshold": row["threshold"],
        "mAP@0.50": row.get("mAP@0.50"),
        "mAP@0.5:0.95": row.get("mAP@0.5:0.95"),
        "n_tiles": n_tiles,
        "p_plus": p_plus,
        "gate_gflops_per_tile": g_gate,
        "detector_gflops_per_tile": g_det,
        "g_overhead": g,
        "accept_rate": accept,
        "tpr": tpr,
        "fpr": fpr,
        # Budget split, normalised to the detector-only baseline. The first four
        # terms sum to 1; gate overhead is additive on top.
        "term_detector_on_true_positives": p_plus * tpr,
        "term_positives_dropped": p_plus * (1.0 - tpr),
        "term_false_positive_leak": (1.0 - p_plus) * fpr,
        "term_correctly_skipped": (1.0 - p_plus) * (1.0 - fpr),
        "term_gate_overhead": g,
        "saved_detector_only": saved_detector_only,
        "saved_vs_full_pass": saved_repo,
        "ceiling_detector_only": 1.0 - p_plus - g,
        "ceiling_vs_full_pass": (1.0 - p_plus) / (1.0 + g),
        "identity_residual_gflops": residual,
        "source": row["_source"],
    }


def main() -> int:
    out_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    worst_residual = 0.0

    for domain in DOMAINS + sweep_domains() + oan_domains():
        if "p_plus_override" in domain:
            p_plus = domain["p_plus_override"]
            n_pos = n_tiles = 0
        else:
            p_plus, n_pos, n_tiles = positive_rate(
                domain["metadata"], domain["split"], domain["classes"]
            )
        rows = load_rows(domain["globs"])
        real = [r for r in rows if r.get("gate") != "oracle"]

        decomposed = []
        for row in real:
            item = decompose(row, p_plus)
            if item is None:
                continue
            item["domain"] = domain["name"]
            item["detector"] = domain["detector"]
            decomposed.append(item)
            out_rows.append(item)
            worst_residual = max(worst_residual, abs(item["identity_residual_gflops"]))

        # Headline operating point: the cheapest row still meeting the mAP
        # tolerance. Two tolerances are carried deliberately, because the draft
        # and the code that produced its numbers disagree:
        #   "absolute" -- within 3 pp of full-pass mAP@0.5, which is what
        #                 paper/main.tex states throughout;
        #   "relative" -- at least 97 % of full-pass mAP@0.5, which is what
        #                 scripts/sparsity_ceiling.py:43 actually implements and
        #                 what the published numbers were computed with.
        # They coincide only when full-pass mAP is exactly 1.0; on small-vehicle
        # (mAP 0.464) the relative rule is a 1.4 pp budget, less than half the
        # stated one.
        full_pass = next((r for r in real if r["threshold"] == 0.0), None)
        best: dict[str, Any] | None = None
        best_by_rule: dict[str, dict[str, Any] | None] = {"absolute": None, "relative": None}
        if full_pass and decomposed:
            tolerances = {
                "absolute": full_pass["mAP@0.50"] - 0.03,
                "relative": full_pass["mAP@0.50"] * 0.97,
            }
            for rule, floor in tolerances.items():
                candidates = [d for d in decomposed if d["mAP@0.50"] >= floor]
                if candidates:
                    best_by_rule[rule] = min(
                        candidates, key=lambda d: d["g_overhead"] + d["accept_rate"]
                    )
            best = best_by_rule["absolute"]

        summary.append({
            "domain": domain["name"],
            "detector": domain["detector"],
            "split": domain["split"],
            "p_plus": p_plus,
            "n_positive_tiles": n_pos,
            "n_tiles": n_tiles,
            "n_operating_points": len(decomposed),
            "full_pass_mAP@0.50": full_pass["mAP@0.50"] if full_pass else None,
            "best_gate": best["label"] if best else None,
            "best_calibration": best["calibration"] if best else None,
            "best_mAP@0.50": best["mAP@0.50"] if best else None,
            "best_saved_vs_full_pass": best["saved_vs_full_pass"] if best else None,
            "best_saved_detector_only": best["saved_detector_only"] if best else None,
            "ceiling_vs_full_pass": best["ceiling_vs_full_pass"] if best else (1.0 - p_plus),
            "best_tpr": best["tpr"] if best else None,
            "best_fpr": best["fpr"] if best else None,
            "by_tolerance_rule": {
                rule: None if b is None else {
                    "gate": b["label"],
                    "calibration": b["calibration"],
                    "mAP@0.50": b["mAP@0.50"],
                    "saved_vs_full_pass": b["saved_vs_full_pass"],
                    "saved_detector_only": b["saved_detector_only"],
                    "tpr": b["tpr"],
                    "fpr": b["fpr"],
                }
                for rule, b in best_by_rule.items()
            },
        })

    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    (out / "savings_decomposition.json").write_text(json.dumps(out_rows, indent=2))
    (out / "savings_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{'domain':<24}{'p+':>8}{'pts':>6}{'save(3pp)':>11}{'save(97%)':>11}"
          f"{'ceiling':>10}{'TPR':>7}{'FPR':>7}  best gate/calib")
    for s in summary:
        rules = s["by_tolerance_rule"]
        def pct(rule: str) -> str:
            b = rules[rule]
            return f"{100*b['saved_vs_full_pass']:.1f}%" if b else "n/a"
        ceil_ = f"{100*s['ceiling_vs_full_pass']:.1f}%"
        tpr = f"{s['best_tpr']:.3f}" if s["best_tpr"] is not None else "n/a"
        fpr = f"{s['best_fpr']:.3f}" if s["best_fpr"] is not None else "n/a"
        print(f"{s['domain']:<24}{s['p_plus']:>8.4f}{s['n_operating_points']:>6}"
              f"{pct('absolute'):>11}{pct('relative'):>11}{ceil_:>10}{tpr:>7}{fpr:>7}"
              f"  {s['best_gate']}/{s['best_calibration']}")

    print(f"\n  {len(out_rows)} operating points decomposed")
    print(f"  worst identity residual: {worst_residual:.3e} GFLOPs")
    if worst_residual > 1e-6:
        print("  FAIL: identity does not reproduce recorded totals")
        return 1
    print("  identity reproduces every recorded total to floating-point precision")
    print(f"  wrote {out}/savings_decomposition.json and savings_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
