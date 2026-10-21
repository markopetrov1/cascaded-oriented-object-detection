#!/usr/bin/env python3
"""Bisect why scripts/train_codesign.py produces a gate that ranks tiles backwards (F3).

The original shared-backbone run, and the script rerun for one epoch on CPU and
on GPU, score validation tiles below chance while their training loss falls. A
minimal loop with the same model, data and sampler learns normally. This runs
the minimal loop for one full epoch with the loss assembled either as plain BCE
on the gate logit ("plain") or exactly as the script does it, through
src.codesign.shared_backbone_loss with the OBB term weighted 0 ("script"),
and records validation ROC-AUC and AP on a fixed 600-tile subset every 100 steps.

    python scripts/r1_codesign_bisect.py --loss plain --device 0
    python scripts/r1_codesign_bisect.py --loss script --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.codesign import SharedBackboneCascade, shared_backbone_loss  # noqa: E402
from src.gate import make_balanced_sampler  # noqa: E402
from train_codesign import CodesignDataset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--loss", choices=("plain", "script"), required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--steps", type=int, default=492)
    p.add_argument("--every", type=int, default=100)
    args = p.parse_args()
    dev = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.manual_seed(67); np.random.seed(67)
    root = REPO / "data/processed/dota_ships"
    tr = CodesignDataset(root, "train", image_size=256, augment=True, feat_size=8)
    va = CodesignDataset(root, "val", image_size=256, augment=False, feat_size=8)
    rng = np.random.default_rng(0)
    lab = va.labels
    vidx = rng.permutation(np.concatenate([rng.choice(np.nonzero(lab == 1)[0], 150, replace=False),
                                           rng.choice(np.nonzero(lab == 0)[0], 450, replace=False)]))
    vl = DataLoader(Subset(va, vidx), batch_size=64, num_workers=6)
    dl = DataLoader(tr, batch_size=32, sampler=make_balanced_sampler(tr.labels), num_workers=8, drop_last=True)
    model = SharedBackboneCascade("resnet18", pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def evaluate():
        model.eval()
        P, Y = [], []
        with torch.no_grad():
            for b in vl:
                P.append(model(b["image"].to(dev))["gate_logit"].float().cpu())
                Y.append(b["label"])
        model.train()
        P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
        return float(roc_auc_score(Y, P)), float(average_precision_score(Y, P))

    hist, run, t0 = [], [], time.time()
    model.train()
    for k, b in enumerate(dl):
        x, y = b["image"].to(dev), b["label"].to(dev)
        out = model(x)
        if args.loss == "plain":
            loss = F.binary_cross_entropy_with_logits(out["gate_logit"], y)
        else:
            tg = {"cls": b["obb_cls"].to(dev), "box": b["obb_box"].to(dev), "angle": b["obb_angle"].to(dev)}
            loss = shared_backbone_loss(out, y, tg, b["obb_mask"].to(dev), gate_weight=1.0, obb_weight=0.0)["total"]
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        run.append(float(loss))
        if (k + 1) % args.every == 0 or k + 1 == args.steps:
            roc, ap = evaluate()
            hist.append({"step": k + 1, "train_loss": float(np.mean(run[-args.every:])), "val_roc": roc, "val_ap": ap})
            print(args.loss, hist[-1], f"{time.time() - t0:.0f}s", flush=True)
        if k + 1 >= args.steps:
            break
    dest = REPO / f"reports/r1/codesign_bisect_{args.loss}.json"
    dest.write_text(json.dumps(hist, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
