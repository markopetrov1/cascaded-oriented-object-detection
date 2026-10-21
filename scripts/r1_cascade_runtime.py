#!/usr/bin/env python3
"""End-to-end timing of the full pass, the decoupled cascade and the fused cascade (R1.2, R2.1).

Every pipeline processes the same tiles of the same DOTA val scenes on the same
card, from pinned host memory to merged image-level detections:

  full       host->device, normalise, pad to 1024, detector, rotated NMS, image merge
  decoupled  host->device, resize to 256 + ImageNet normalise, gate, threshold;
             detector, NMS and merge on the accepted tiles only
  fused      host->device, normalise, pad, backbone layers 0..10 on every tile, OAN
             head (max over its grid), threshold; layers 11..23, NMS and merge on
             the accepted tiles only, reusing the backbone features the neck needs

Operating points come from scripts/r1_eval_cascade.py (risk rule, selected on all
of DOTA val), so the two cascades run at matched accuracy. Detection uses the
deployment confidence 0.25 and max_det 1000 (the evaluation caches use 0.001 so
that AP is not truncated, which is an evaluation concern, not a deployment one).
Image merging uses class-wise rotated NMS at IoU 0.3 with probIoU, as the
Ultralytics DOTA merge does.

Configurations: "streaming" (every stage at batch 1) and "batched" (gate over a
scene's tiles at batch 32, detector and backbone at batch 16). Each is timed
three times after a warm-up pass and the median total is reported, with a
per-stage breakdown from CUDA events. The cost identity's prediction from the
separately measured component latencies (reports/r1/speed/components.json) is
printed next to each measurement.

Writes reports/r1/speed/runtime.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402

configure_ultralytics_settings()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import nms as unms  # noqa: E402
from ultralytics.utils.metrics import batch_probiou  # noqa: E402

from src.gate import build_gate_model  # noqa: E402
from src.r1_eval import parse_stem  # noqa: E402

TILES = REPO / "data/processed/dota_ships/images/val"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TAP = 10


class Timer:
    """Per-stage time. With sync=False it does nothing, so timed repeats carry no
    profiling cost. With sync=True every stage is bracketed by device
    synchronisation and wall-clock reads, which serialises the pipeline and so
    is used only in a separate profiling pass. (CUDA events around the fused
    prefix under-reported it in the first run, while the synchronised totals
    agreed with the component benchmark; synchronised stages avoid the issue.)"""

    def __init__(self, sync: bool = False):
        self.sync = sync
        self.ms = defaultdict(float)

    def __call__(self, stage):
        timer = self

        class _Ctx:
            def __enter__(self_):
                if timer.sync:
                    torch.cuda.synchronize(); self_.t0 = time.perf_counter()

            def __exit__(self_, *a):
                if timer.sync:
                    torch.cuda.synchronize()
                    timer.ms[stage] += (time.perf_counter() - self_.t0) * 1000
        return _Ctx()

    def totals(self) -> dict:
        return dict(self.ms)


def to_det_input(batch_u8: torch.Tensor) -> torch.Tensor:
    x = batch_u8.permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[2:]
    if h != 1024 or w != 1024:
        x = F.pad(x, (0, 1024 - w, 0, 1024 - h), value=114 / 255)
    return x


def to_gate_input(batch_u8: torch.Tensor, dev) -> torch.Tensor:
    x = batch_u8.permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[2:]
    s = 256 / max(h, w)
    x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False)
    x = F.pad(x, (0, 256 - x.shape[3], 0, 256 - x.shape[2]), value=0.0)
    return (x - MEAN.to(dev)) / STD.to(dev)


def postprocess(preds, nc: int, conf: float, max_det: int):
    return unms.non_max_suppression(preds, conf, 0.7, max_det=max_det, nc=nc, rotated=True)


def merge(dets: list[tuple[torch.Tensor, int, int]]) -> torch.Tensor:
    """Image merge: shift tile detections and run class-wise rotated NMS at IoU 0.3."""
    if not dets:
        return torch.empty(0, 7)
    parts = []
    for d, x0, y0 in dets:
        if len(d):
            d = d.clone(); d[:, 0] += x0; d[:, 1] += y0
            parts.append(d)
    if not parts:
        return torch.empty(0, 7)
    d = torch.cat(parts)
    c = d[:, 5:6] * 7680.0
    boxes = torch.cat((d[:, :2] + c, d[:, 2:4], d[:, 6:7]), dim=-1)
    keep = unms.TorchNMS.fast_nms(boxes, d[:, 4], 0.3, iou_func=batch_probiou)
    return d[keep]


def run_suffix(model, y: list, idx: torch.Tensor):
    yy = [t[idx] if t is not None else None for t in y]
    o = yy[TAP]
    for m in model.model[TAP + 1:]:
        if m.f != -1:
            o = yy[m.f] if isinstance(m.f, int) else [o if j == -1 else yy[j] for j in m.f]
        o = m(o)
        yy.append(o if m.i in model.save else None)
    return o


def run_prefix(model, x):
    y, o = [], x
    for m in model.model[:TAP + 1]:
        o = m(o)
        y.append(o if (m.i in model.save or m.i == TAP) else None)
    return y


def pipeline(kind: str, scenes, models: dict, cfg: dict, dev, timer: Timer) -> dict:
    """Run one pipeline over all scenes; returns counts. Timing is done by the caller."""
    n_tiles = n_acc = 0
    merged = 0
    bg, bd = cfg["gate_batch"], cfg["det_batch"]
    for name, tiles in scenes:
        per_tile = []
        if kind == "full":
            for i in range(0, len(tiles), bd):
                chunk = tiles[i:i + bd]
                with timer("transfer+preprocess"):
                    x = to_det_input(torch.stack([t for _, t in chunk]).to(dev, non_blocking=True))
                with timer("detector"):
                    preds = models["det"](x)
                with timer("nms"):
                    out = postprocess(preds, models["nc"], cfg["conf"], cfg["max_det"])
                per_tile += [(o, *offs) for o, (offs, _) in zip(out, chunk)]
            n_tiles += len(tiles); n_acc += len(tiles)
        elif kind == "decoupled":
            accepted = []
            for i in range(0, len(tiles), bg):
                chunk = tiles[i:i + bg]
                with timer("transfer+preprocess"):
                    u8 = torch.stack([t for _, t in chunk]).to(dev, non_blocking=True)
                    g = to_gate_input(u8, dev)
                with timer("gate"):
                    p = torch.sigmoid(models["gate"](g).squeeze(-1))
                acc = (p >= cfg["threshold"]).cpu().numpy()
                accepted += [(chunk[k], u8[k]) for k in np.nonzero(acc)[0]]
            for i in range(0, len(accepted), bd):
                chunk = accepted[i:i + bd]
                with timer("transfer+preprocess"):
                    x = to_det_input(torch.stack([u for _, u in chunk]))
                with timer("detector"):
                    preds = models["det"](x)
                with timer("nms"):
                    out = postprocess(preds, models["nc"], cfg["conf"], cfg["max_det"])
                per_tile += [(o, *c[0][0]) for o, c in zip(out, chunk)]
            n_tiles += len(tiles); n_acc += len(accepted)
        elif kind == "fused":
            for i in range(0, len(tiles), bd):
                chunk = tiles[i:i + bd]
                with timer("transfer+preprocess"):
                    x = to_det_input(torch.stack([t for _, t in chunk]).to(dev, non_blocking=True))
                with timer("backbone"):
                    y = run_prefix(models["det"], x)
                with timer("gate"):
                    p = torch.sigmoid(models["head"](y[TAP])).amax(dim=(1, 2, 3))
                idx = torch.nonzero(p >= cfg["threshold"]).squeeze(1)
                if len(idx):
                    with timer("neck+head"):
                        preds = run_suffix(models["det"], y, idx)
                    with timer("nms"):
                        out = postprocess(preds, models["nc"], cfg["conf"], cfg["max_det"])
                    sel = idx.cpu().tolist()
                    per_tile += [(o, *chunk[k][0]) for o, k in zip(out, sel)]
                n_tiles += len(chunk); n_acc += len(idx)
        with timer("merge"):
            m = merge(per_tile)
        merged += len(m)
    return {"n_tiles": n_tiles, "n_accepted": n_acc, "merged_detections": merged}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0")
    p.add_argument("--images", type=int, default=60, help="number of DOTA val scenes (fold B, every k-th)")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", default="reports/r1/speed/runtime.json")
    args = p.parse_args()
    dev = torch.device(f"cuda:{args.device}")
    torch.backends.cudnn.benchmark = False

    folds = yaml.safe_load(open(REPO / "configs/splits/r1_val_2fold.yaml"))
    imgs = folds["B"][:: max(1, len(folds["B"]) // args.images)][: args.images]
    by_img = defaultdict(list)
    for f in sorted(TILES.glob("*.png")):
        src, x, y, w, h = parse_stem(f.stem)
        if src in imgs:
            arr = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            by_img[src].append(((x, y), torch.from_numpy(arr).pin_memory()))
    scenes = [(k, by_img[k]) for k in imgs if k in by_img]
    n = sum(len(t) for _, t in scenes)
    print(f"[runtime] {len(scenes)} scenes, {n} tiles preloaded in pinned memory", flush=True)

    indep = json.load(open(REPO / "reports/r1/cascade/indep/ship.json"))["selections"]["risk"]["all"]
    fused = json.load(open(REPO / "reports/r1/cascade/fused/ship.json"))["selections"]["risk"]["all"]
    det = YOLO(str(REPO / "runs/obb/runs/baseline_yolo11m_dota_ships_obb_p50/weights/best.pt"), task="obb").model.float().to(dev).eval()
    oan = YOLO(str(REPO / "runs/obb/runs/oan/oan_ships_lambda3/weights/best.pt"), task="obb").model.float().to(dev).eval()
    gate = build_gate_model("mobilenetv3_large_100", pretrained=False)
    gate.load_state_dict(torch.load(REPO / "runs/gate_mbv3large/best.pt", map_location="cpu", weights_only=False)["model"])
    gate = gate.to(dev).eval()
    models = {"full": {"det": det, "nc": 1}, "decoupled": {"det": det, "gate": gate, "nc": 1},
              "fused": {"det": oan, "head": oan.oan_head, "nc": 1}}
    thresholds = {"full": None, "decoupled": indep["threshold"], "fused": fused["threshold"]}

    out = {"scenes": len(scenes), "tiles": n, "images": imgs, "operating_points": {"decoupled": indep, "fused": fused},
           "conf": 0.25, "max_det": 1000, "merge": "class-wise rotated NMS, probIoU, IoU 0.3", "runs": {}}
    configs = {"streaming": {"gate_batch": 1, "det_batch": 1}, "batched": {"gate_batch": 32, "det_batch": 16}}
    with torch.inference_mode():
        for cname, c in configs.items():
            for kind in ("full", "decoupled", "fused"):
                cfg = dict(c, conf=0.25, max_det=1000, threshold=thresholds[kind])
                # Warm-up over every scene: accepted-tile batches take many shapes, and a short
                # warm-up left the first timed repeat several times slower than the rest.
                pipeline(kind, scenes, models[kind], cfg, dev, Timer())
                totals, counts = [], None
                for _ in range(args.repeats):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    counts = pipeline(kind, scenes, models[kind], cfg, dev, Timer())
                    torch.cuda.synchronize()
                    totals.append((time.perf_counter() - t0) * 1000)
                med = statistics.median(totals)
                prof = Timer(sync=True)                 # one separate, serialised profiling pass
                torch.cuda.synchronize(); t0 = time.perf_counter()
                pipeline(kind, scenes, models[kind], cfg, dev, prof)
                torch.cuda.synchronize()
                prof_total = (time.perf_counter() - t0) * 1000
                st = prof.totals()
                st["unattributed"] = prof_total - sum(st.values())
                out["runs"][f"{cname}/{kind}"] = dict(counts, total_ms=med, totals_ms=totals,
                                                      ms_per_tile=med / counts["n_tiles"], stage_ms=st,
                                                      profiled_total_ms=prof_total,
                                                      accept_rate=counts["n_accepted"] / counts["n_tiles"])
                print(f"{cname:<9} {kind:<9} accept={counts['n_accepted'] / counts['n_tiles']:.3f} "
                      f"total={med:8.0f} ms  per tile={med / counts['n_tiles']:.2f} ms  "
                      + " ".join(f"{k}={v:.0f}" for k, v in st.items()), flush=True)
    for cname in configs:
        f = out["runs"][f"{cname}/full"]["total_ms"]
        for kind in ("decoupled", "fused"):
            r = out["runs"][f"{cname}/{kind}"]
            r["time_saved_vs_full"] = 1 - r["total_ms"] / f
    Path(REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.out).write_text(json.dumps(out, indent=2, default=float))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
