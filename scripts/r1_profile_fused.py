#!/usr/bin/env python3
"""Profile the fused cascade path to locate time outside its timed stages.

The synchronised stage profile of scripts/r1_cascade_runtime.py leaves several
seconds of the fused streaming run unattributed, although its total agrees with
the component costs. This runs the fused pipeline on a few scenes under
torch.profiler and writes the top operators by CPU and by CUDA time, plus the
wall time of each Python-level step of the loop measured with synchronisation,
so the gap can be traced to an operator or to the loop itself.

    python scripts/r1_profile_fused.py --device 1 --scenes 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import r1_cascade_runtime as rt  # noqa: E402  (configures Ultralytics on import)
import cv2  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.r1_eval import parse_stem  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0")
    p.add_argument("--scenes", type=int, default=6)
    args = p.parse_args()
    dev = torch.device(f"cuda:{args.device}")
    # Make it the current device: torch.cuda.synchronize() and CUDA events otherwise refer to
    # cuda:0 and do not wait for, or time, work running on another card.
    torch.cuda.set_device(dev)
    folds = yaml.safe_load(open(REPO / "configs/splits/r1_val_2fold.yaml"))
    imgs = folds["B"][:: max(1, len(folds["B"]) // 60)][: args.scenes]
    by_img = defaultdict(list)
    for f in sorted(rt.TILES.glob("*.png")):
        src, x, y, w, h = parse_stem(f.stem)
        if src in imgs:
            by_img[src].append(((x, y), torch.from_numpy(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)).pin_memory()))
    scenes = [(k, by_img[k]) for k in imgs if k in by_img]
    oan = YOLO(str(REPO / "runs/obb/runs/oan/oan_ships_lambda3/weights/best.pt"), task="obb").model.float().to(dev).eval()
    thr = json.load(open(REPO / "reports/r1/cascade/fused/ship.json"))["selections"]["risk"]["all"]["threshold"]
    steps = defaultdict(float)

    def step(name, fn):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize(); steps[name] += (time.perf_counter() - t0) * 1000
        return r

    with torch.inference_mode():
        for _ in range(2):                                    # warm-up
            for _, tiles in scenes:
                for offs, t in tiles:
                    x = rt.to_det_input(t[None].to(dev))
                    y = rt.run_prefix(oan, x)
        steps.clear()
        n = 0
        torch.cuda.synchronize(); t_all = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _, tiles in scenes:
                for offs, t in tiles:
                    n += 1
                    x = step("to_det_input", lambda: rt.to_det_input(t[None].to(dev, non_blocking=True)))
                    y = step("run_prefix", lambda: rt.run_prefix(oan, x))
                    pr = step("head", lambda: torch.sigmoid(oan.oan_head(y[rt.TAP])).amax(dim=(1, 2, 3)))
                    idx = step("nonzero", lambda: torch.nonzero(pr >= thr).squeeze(1))
                    if len(idx):
                        preds = step("run_suffix", lambda: rt.run_suffix(oan, y, idx))
                        step("nms", lambda: rt.postprocess(preds, 1, 0.25, 1000))
        torch.cuda.synchronize(); total = (time.perf_counter() - t_all) * 1000
    out = {"tiles": n, "total_ms": total, "steps_ms": dict(steps), "unattributed_ms": total - sum(steps.values()),
           "top_cpu": prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15),
           "top_cuda": prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)}
    dest = REPO / "reports/r1/speed/profile_fused.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("top")}, indent=2))
    print(out["top_cpu"])
    print(out["top_cuda"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
