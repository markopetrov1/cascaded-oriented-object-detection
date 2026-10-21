#!/usr/bin/env python3
"""Where should a fused gate tap the detector? (WP9; R1.2, R2.1)

A gate that reads the detector's own features is not free: every tile pays the
backbone prefix up to the tap before the gate can decide. Tapping earlier makes
the prefix cheaper but gives the gate weaker features. This script measures that
trade-off on the 50-epoch ships detector with the trunk frozen, so the detector
and its accuracy stay exactly as they are and only the gate changes.

For each tap k in {2, 4, 6, 8, 10} an OAN-style head (3x3 s2 conv to 256, 1x1 to
512, 1x1 to 1; src/oan.py) is trained on the tap's feature map after average
pooling to 32x32, so every head sees the same 16x16 output grid as OAN's at layer
10 and has the same capacity; only the tap differs. Grid cells are positive when
a ship centre falls inside them, the loss is OAN's focal loss, and a tile's score
is the maximum cell probability. All five heads train in one pass over the
frozen trunk, for a fixed number of epochs with no checkpoint selection on val.

Writes reports/r1/tap_scores/tap{k}_seed{s}_val.jsonl (same schema as the other
gate score files), the head weights under runs/r1_tap/, and the measured
per-tap costs to reports/r1/tap_scores/costs_seed{s}.json.

    python scripts/r1_tap_study.py --device 0 --epochs 12 --seed 67
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402

configure_ultralytics_settings()

import cv2  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from thop import profile  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.oan import OANHead, oan_focal_loss  # noqa: E402

TAPS = [2, 4, 6, 8, 10]
POOL_TO = 32          # every tap is pooled to 32x32, the stride-32 map at 1024 px
DETECTOR = "runs/obb/runs/baseline_yolo11m_dota_ships_obb_p50/weights/best.pt"
DATA = REPO / "data/processed/dota_ships"


class Tiles(Dataset):
    """Tile images resized to imgsz (as scripts/score_tiles_oan.py does) with a 16x16 centre grid."""

    def __init__(self, split: str, imgsz: int, flip: bool, limit: int = 0):
        self.paths = sorted((DATA / "images" / split).glob("*.png"))
        if limit:
            self.paths = self.paths[::max(1, len(self.paths) // limit)][:limit]
        self.labels = DATA / "labels" / split
        self.gate = DATA / "gate_labels" / split
        self.imgsz, self.flip = imgsz, flip

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        im = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        grid = np.zeros((1, 16, 16), np.float32)
        lf = self.labels / f"{p.stem}.txt"
        if lf.exists():
            for line in lf.read_text().splitlines():
                v = line.split()
                if len(v) >= 9:
                    xy = np.array(v[1:9], float).reshape(4, 2).mean(0)
                    gx, gy = (np.clip(xy, 0, 1 - 1e-6) * 16).astype(int)
                    grid[0, gy, gx] = 1.0
        if self.flip and np.random.rand() < 0.5:
            im, grid = im[:, ::-1], grid[:, :, ::-1]
        if self.flip and np.random.rand() < 0.5:
            im, grid = im[::-1], grid[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
        gl = self.gate / f"{p.stem}.txt"
        label = int(gl.read_text().strip() or "0") if gl.exists() else 0
        return x, torch.from_numpy(np.ascontiguousarray(grid)), label, p.stem


class TapHead(nn.Module):
    def __init__(self, ch: int, size: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(POOL_TO) if size != POOL_TO else nn.Identity()
        self.head = OANHead(ch)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(f))


def trunk_features(layers: nn.ModuleList, x: torch.Tensor) -> dict[int, torch.Tensor]:
    out = {}
    for i, m in enumerate(layers[:max(TAPS) + 1]):
        x = m(x)
        if i in TAPS:
            out[i] = x
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="smoke test on this many tiles per split")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    amp = dict(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    det = YOLO(str(REPO / DETECTOR), task="obb").model.float().to(dev).eval()
    for q in det.parameters():
        q.requires_grad_(False)
    layers = det.model
    assert all(m.f == -1 for m in layers[:max(TAPS) + 1])

    with torch.no_grad():
        feats = trunk_features(layers, torch.zeros(1, 3, args.imgsz, args.imgsz, device=dev))
    heads = nn.ModuleDict({str(k): TapHead(int(f.shape[1]), int(f.shape[-1])) for k, f in feats.items()}).to(dev)
    costs = {}
    for k, f in feats.items():
        pre = nn.Sequential(*layers[:k + 1])
        macs_p, _ = profile(copy.deepcopy(pre).cpu(), inputs=(torch.zeros(1, 3, args.imgsz, args.imgsz),), verbose=False)
        macs_h, _ = profile(copy.deepcopy(heads[str(k)]).cpu(), inputs=(f.detach().cpu(),), verbose=False)
        costs[k] = {"prefix_gflops": 2 * macs_p / 1e9, "head_gflops": 2 * macs_h / 1e9,
                    "feature_shape": list(f.shape[1:])}
    macs_d, _ = profile(copy.deepcopy(det).cpu(), inputs=(torch.zeros(1, 3, args.imgsz, args.imgsz),), verbose=False)
    det_gflops = 2 * macs_d / 1e9
    print({k: (round(v["prefix_gflops"], 2), round(v["head_gflops"], 3)) for k, v in costs.items()},
          "detector", round(det_gflops, 2), flush=True)

    train = DataLoader(Tiles("train", args.imgsz, flip=True, limit=args.limit), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train),
                                                pct_start=0.1)
    tag = f"seed{args.seed}" + ("_smoke" if args.limit else "")
    run_dir = REPO / f"runs/r1_tap/{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(1, args.epochs + 1):
        heads.train(); t0 = time.time(); tot = {k: 0.0 for k in TAPS}; n = 0
        for x, grid, _, _ in train:
            x, grid = x.to(dev, non_blocking=True), grid.to(dev, non_blocking=True)
            with torch.no_grad(), torch.autocast(**amp):
                fs = trunk_features(layers, x)
            loss = 0.0
            for k in TAPS:
                lk = oan_focal_loss(heads[str(k)](fs[k].float()), grid)
                tot[k] += float(lk.detach()); loss = loss + lk
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); n += 1
        row = {"epoch": ep, "seconds": round(time.time() - t0, 1), **{f"loss_tap{k}": tot[k] / n for k in TAPS}}
        history.append(row); print(row, flush=True)
        torch.save({"heads": heads.state_dict(), "args": vars(args), "epoch": ep}, run_dir / "last.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    out_dir = REPO / ("reports/r1/tap_scores" + ("_smoke" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    val = DataLoader(Tiles("val", args.imgsz, flip=False, limit=args.limit), batch_size=args.batch, shuffle=False,
                     num_workers=args.workers, pin_memory=True)
    heads.eval()
    fh = {k: open(out_dir / f"tap{k}_seed{args.seed}_val.jsonl", "w") for k in TAPS}
    with torch.no_grad():
        for x, _, label, stem in val:
            with torch.autocast(**amp):
                fs = trunk_features(layers, x.to(dev))
            for k in TAPS:
                prob = heads[str(k)](fs[k].float()).sigmoid().amax(dim=(1, 2, 3)).cpu()
                for s, pr, lb in zip(stem, prob, label):
                    fh[k].write(json.dumps({"tile_id": s, "prob": float(pr), "label": int(lb),
                                            "split": "val"}) + "\n")
    for f in fh.values():
        f.close()
    (out_dir / f"costs_seed{args.seed}.json").write_text(json.dumps(
        {"detector": DETECTOR, "detector_gflops": det_gflops, "pool_to": POOL_TO, "taps": costs,
         "epochs": args.epochs, "history": history}, indent=2))
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
