#!/usr/bin/env python3
"""Detector predictions and gate scores on SODA-A validation tiles, tiled in memory.

Tiles are cut from each scene with the windows in data/processed/soda_a/metadata/
tiles.jsonl and never written to disk. The detector output uses the same label
layout as scripts/r1_predict.py (conf 0.001, max_det 10000, coordinates
normalised to the tile), and gate scores the same JSONL schema as
scripts/score_tiles.py, with the gate's eval transform (longest side to 256,
zero padding, ImageNet normalisation, RGB). Nothing here is trained or tuned on
SODA-A.

    python scripts/r1_soda_infer.py --device 0 --detector runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/weights/best.pt \
        --det-out runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/predict_soda_val_c001 \
        --gate ship=runs/gate_sweep_ship/best.pt ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402

configure_ultralytics_settings()

import cv2  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.gate import build_gate_model  # noqa: E402

META = REPO / "data/processed/soda_a/metadata/tiles.jsonl"
IMAGES = REPO / "data/raw/SODA-A/Images/val"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def gate_input(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """Same as GateDataset's eval transform: LongestMaxSize, PadIfNeeded(0), Normalize."""
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR) if s != 1 else rgb
    # PadIfNeeded (centred, constant 0) runs before Normalize, so the border is -mean/std
    out = np.broadcast_to(-MEAN / STD, (size, size, 3)).copy()
    top, left = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
    out[top:top + im.shape[0], left:left + im.shape[1]] = (im.astype(np.float32) / 255.0 - MEAN) / STD
    return out.transpose(2, 0, 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0")
    p.add_argument("--detector", default=None)
    p.add_argument("--det-out", default=None)
    p.add_argument("--gate", action="append", default=[], help="NAME=checkpoint (repeatable)")
    p.add_argument("--gate-out", default="reports/r1/soda/gate_scores")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    dev = torch.device(f"cuda:{args.device}")
    # Make it the current device: torch.cuda.synchronize() and CUDA events otherwise refer to
    # cuda:0 and do not wait for, or time, work running on another card.
    torch.cuda.set_device(dev)
    by_img = defaultdict(list)
    for line in open(META):
        r = json.loads(line)
        by_img[r["source_image"]].append(r)

    det = YOLO(str(REPO / args.detector), task="obb") if args.detector else None
    if det:
        labels = REPO / args.det_out / "labels"
        labels.mkdir(parents=True, exist_ok=True)
    gates = {}
    for g in args.gate:
        name, ck = g.split("=", 1)
        c = torch.load(REPO / ck, map_location="cpu", weights_only=False)
        m = build_gate_model(c.get("config", {}).get("backbone"), pretrained=False)
        m.load_state_dict(c["model"])
        gates[name] = m.to(dev).eval()
    gout = REPO / args.gate_out
    gout.mkdir(parents=True, exist_ok=True)
    gfh = {n: open(gout / f"{n}.jsonl", "w") for n in gates}
    t0, n_det, n_tiles = time.time(), 0, 0
    for k, (img_name, rows) in enumerate(sorted(by_img.items())):
        bgr = cv2.imread(str(IMAGES / img_name), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"missing image {img_name}")
        crops = [np.ascontiguousarray(bgr[r["y"]:r["y"] + r["height"], r["x"]:r["x"] + r["width"]]) for r in rows]
        if det:
            for i in range(0, len(crops), args.batch):
                res = det.predict(crops[i:i + args.batch], imgsz=1024, conf=0.001, iou=0.7, max_det=10000,
                                  device=args.device, verbose=False, save=False)
                for r, out in zip(rows[i:i + args.batch], res):
                    lines = []
                    obb = out.obb
                    if obb is not None and len(obb):
                        xy = obb.xyxyxyxyn.cpu().numpy().reshape(-1, 8)
                        cl = obb.cls.cpu().numpy().astype(int)
                        cf = obb.conf.cpu().numpy()
                        lines = [f"{c} " + " ".join(f"{v:.6f}" for v in q) + f" {s:.6f}" for c, q, s in zip(cl, xy, cf)]
                        n_det += len(lines)
                    (labels / f"{r['tile_id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if gates:
            x = torch.from_numpy(np.stack([gate_input(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops])).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for n, m in gates.items():
                    pr = torch.sigmoid(m(x).float().squeeze(-1)).cpu().numpy()
                    for r, v in zip(rows, pr):
                        gfh[n].write(json.dumps({"tile_id": r["tile_id"], "prob": float(v), "split": "val"}) + "\n")
        n_tiles += len(rows)
        if k % 50 == 0:
            print(f"  {k + 1}/{len(by_img)} scenes, {n_tiles} tiles, {time.time() - t0:.0f}s", flush=True)
    for f in gfh.values():
        f.close()
    if det:
        (REPO / args.det_out / "r1_predict_meta.json").write_text(json.dumps(
            {"weights": args.detector, "source": "SODA-A val, tiled in memory (1024/200)", "conf": 0.001,
             "iou": 0.7, "max_det": 10000, "imgsz": 1024, "tiles": n_tiles, "detections": n_det,
             "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"done: {n_tiles} tiles, {n_det} detections, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
