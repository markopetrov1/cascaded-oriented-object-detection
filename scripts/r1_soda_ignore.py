#!/usr/bin/env python3
"""SODA-A transfer: how much the ceilings depend on the ignore tiles, and how uncertain the
accuracy changes are (second-round review).

SODA-A marks regions whose objects were not annotated as ignore regions. The transfer evaluation
(scripts/r1_eval_cascade.py --preset soda_sweep) counts the tiles touching one towards the accept
rate, and so towards the saving, but not towards p+. For each task, at the MobileNetV3-large sweep
gate's operating point chosen on all of DOTA validation, this script reports

  - the gate's accept rate on ignore tiles and on labelled tiles;
  - the lossless ceiling 1 - p+ with the ignore tiles left out (as reported), counted as empty,
    counted as positive, and counted as positive only when they also hold a labelled target;
  - the saving over the labelled tiles alone, comparable with the reported ceiling;
  - the 95% interval of the AP@0.5 change from 1000 image-level bootstrap resamples.

    python scripts/r1_soda_ignore.py      # writes reports/r1/soda_ignore.json, CPU only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import r1_eval_cascade as E  # noqa: E402
from src.r1_eval import EvalCache  # noqa: E402


def main() -> int:
    tr = E.PRESETS["soda_sweep"]["transfer"]
    cache = EvalCache.load(REPO / tr["cache"])
    meta = {}
    for line in open(REPO / tr["meta"]):
        r = json.loads(line)
        meta[r["tile_id"]] = r
    ign = np.array([bool(meta[t].get("has_ignore")) for t in cache.tiles])
    out = {"note": "MobileNetV3-large sweep gates at the threshold chosen on all of DOTA val (rule risk)",
           "n_tiles": len(cache.tiles), "n_ignore_tiles": int(ign.sum()), "tasks": {}}
    for task in ("ship", "plane", "small vehicle"):
        slug = task.replace(" ", "_")
        run = json.load(open(REPO / f"reports/r1/cascade/soda_sweep/{slug}.json"))
        rep = run["transfer"]["risk"]
        sel = rep["selected"]
        assert sel["gate"] == "mbv3large" and sel["calibration"] == "identity", sel
        classes = E.task_classes(task)
        want = {E.CLASS_ID_INV[c] for c in classes}
        has = np.array([bool(want & set(meta[t].get("class_counts", {}))) for t in cache.tiles])
        raw = E.load_scores(tr["scores"]["mbv3large"].format(slug=slug), cache.tiles)
        acc = raw >= sel["threshold"]
        gate_g, det_g = run["gates"]["mbv3large"]["gflops"], run["detector_gflops"]
        # the masks are not stored for the transfer, so check that they are reproduced
        assert abs(acc.mean() - rep["accept_rate"]) < 1e-12, (acc.mean(), rep["accept_rate"])
        n, lab = len(acc), ~ign
        pos_lab = has & lab
        ceil = {"ignore_left_out": 1 - pos_lab.sum() / lab.sum(),
                "ignore_as_empty": 1 - pos_lab.sum() / n,
                "ignore_as_positive": 1 - (pos_lab.sum() + ign.sum()) / n,
                "ignore_positive_if_target": 1 - has.sum() / n}
        boot = E.bootstrap(cache, acc, classes, gate_g, det_g, "decoupled")
        out["tasks"][slug] = {
            "threshold": sel["threshold"],
            "accept_rate": float(acc.mean()),
            "accept_rate_ignore_tiles": float(acc[ign].mean()),
            "accept_rate_labelled_tiles": float(acc[lab].mean()),
            "ignore_tiles_with_target": int((has & ign).sum()),
            "ceiling": {k: float(v) for k, v in ceil.items()},
            "saving": float(rep["saving"]),
            "saving_labelled_tiles": float(1 - E.cost(gate_g, float(acc[lab].mean()), "decoupled", det_g)),
            "delta_mAP50_coco": float(rep["delta_mAP50_coco"]),
            "bootstrap": boot,
        }
        t = out["tasks"][slug]
        print(f"{slug:<14} accept {t['accept_rate']:.3f} (ignore {t['accept_rate_ignore_tiles']:.3f}, "
              f"labelled {t['accept_rate_labelled_tiles']:.3f}); saving {t['saving']:.3f}, labelled only "
              f"{t['saving_labelled_tiles']:.3f}; ceilings " +
              ", ".join(f"{k} {v:.3f}" for k, v in t["ceiling"].items()) +
              f"; dAP {100 * t['delta_mAP50_coco']:.2f} pp, CI " +
              ", ".join(f"{100 * v:.2f}" for v in boot["delta_mAP50_coco_ci95"]))
    (REPO / "reports/r1/soda_ignore.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
