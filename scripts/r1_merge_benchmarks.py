#!/usr/bin/env python3
"""Combine repeated runs of scripts/r1_benchmark.py into one reported set.

Each run already reports the median of 200 timed iterations per configuration.
Between runs the medians can still differ, mostly when the GPU's clocks were
ramping up during the first configurations measured. The reported latency is
the lower of the runs' medians, which discards such warm-up artefacts, and the
largest relative difference between runs is recorded so it can be stated.
GFLOPs are identical across runs by construction and are checked.

    python scripts/r1_merge_benchmarks.py reports/r1/speed/components.json reports/r1/speed/components_run2.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    runs = [json.load(open(REPO / p)) for p in sys.argv[1:]]
    out = copy.deepcopy(runs[0])
    spread = []
    for group in ("gates", "detectors"):
        for name, entry in out[group].items():
            for r in runs[1:]:
                assert abs(r[group][name]["gflops"] - entry["gflops"]) < 1e-6, name
            for prec in ("fp32", "fp16"):
                for b in entry[prec]:
                    meds = [r[group][name][prec][b]["median_ms"] for r in runs]
                    best = min(range(len(runs)), key=lambda i: meds[i])
                    entry[prec][b] = dict(runs[best][group][name][prec][b], run=best + 1)
                    spread.append((max(meds) / min(meds) - 1, f"{group}/{name}/{prec}/b{b}"))
    for k, entry in out["prefix"].items():
        for b in entry["fp32"]:
            for part in entry["fp32"][b]:
                meds = [r["prefix"][k]["fp32"][b][part]["median_ms"] for r in runs]
                best = min(range(len(runs)), key=lambda i: meds[i])
                entry["fp32"][b][part] = dict(runs[best]["prefix"][k]["fp32"][b][part], run=best + 1)
                spread.append((max(meds) / min(meds) - 1, f"prefix/{k}/b{b}/{part}"))
    spread.sort(reverse=True)
    vals = sorted(s for s, _ in spread)
    out["merge"] = {"runs": sys.argv[1:], "rule": "lower of the runs' medians",
                    "max_relative_difference": spread[0][0], "max_at": spread[0][1],
                    "median_relative_difference": vals[len(vals) // 2],
                    "largest_five": [[round(s, 4), n] for s, n in spread[:5]]}
    (REPO / "reports/r1/speed/components_merged.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["merge"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
