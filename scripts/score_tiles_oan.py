#!/usr/bin/env python3
"""Emit per-tile objectness from a jointly-trained OAN model.

Writes the same JSONL schema as scripts/score_tiles.py so eval_cascade can
sweep an OAN gate and an independently trained gate identically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# See train_oan.py: must precede the ultralytics import chain.
from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402

configure_ultralytics_settings()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.oan import OANOBBModel, oan_statistic_threshold  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="checkpoint from train_oan.py")
    p.add_argument("--data-root", default="data/processed/dota_ships")
    p.add_argument("--split", default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--model-cfg", default="yolo11m-obb.yaml")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if hasattr(state, "state_dict"):          # a pickled nn.Module
        model = state.float()
    else:
        model = OANOBBModel(cfg=args.model_cfg, nc=args.nc, verbose=False)
        model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    root = Path(args.data_root)
    img_dir = root / "images" / args.split
    gate_dir = root / "gate_labels" / args.split
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"no images under {img_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    maps: list[torch.Tensor] = []
    written = 0
    with out_path.open("w") as fh, torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            chunk = paths[i:i + args.batch_size]
            imgs = []
            for q in chunk:
                im = Image.open(q).convert("RGB").resize((args.imgsz, args.imgsz))
                imgs.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0)
            batch = torch.stack(imgs).to(device)
            model.forward(batch)
            logits = model.oan_head(model._oan_feat)
            probs = logits.sigmoid().amax(dim=(1, 2, 3)).cpu()
            for q, prob, lg in zip(chunk, probs, logits.cpu()):
                lbl_file = gate_dir / f"{q.stem}.txt"
                label = int(lbl_file.read_text().strip() or "0") if lbl_file.exists() else 0
                fh.write(json.dumps({
                    "label": label,
                    "logit": float(torch.logit(prob.clamp(1e-6, 1 - 1e-6))),
                    "prob": float(prob),
                    "split": args.split,
                    "tile_id": q.stem,
                }) + "\n")
                written += 1
                if len(maps) < 2000:
                    maps.append(lg)
            if i % (args.batch_size * 50) == 0:
                print(f"  {written}/{len(paths)}", flush=True)

    thr = oan_statistic_threshold(maps)
    thr_path = out_path.with_suffix(".threshold.json")
    thr_path.write_text(json.dumps({
        "rule": "T = (m + v)^2 / k, k=4 (Xie et al., Sci. China Inf. Sci. 2023)",
        "threshold": thr, "n_maps": len(maps),
    }, indent=2))
    print(f"[score_tiles_oan] {written} tiles -> {out_path}")
    print(f"[score_tiles_oan] statistic threshold {thr:.4f} -> {thr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
