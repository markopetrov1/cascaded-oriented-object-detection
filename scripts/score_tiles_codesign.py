#!/usr/bin/env python3
"""Score tiles using a co-design model's gate output (Pillar 3 in cascade).

The 3 codesign variants in ``src/codesign.py`` each expose a different forward
signature; this script normalizes them so downstream ``eval_cascade.py`` can
consume the resulting JSONL exactly like a binary-trained gate's output:

  - SharedBackboneCascade.forward(x) -> {"gate_logit", ...}
  - EarlyExitWrapper.forward(x)      -> {"gate_logit", ...}     (train-time forward)
  - RelaxedGatingCascade.forward(x)  -> {"keep_mask", "gate_logits": (B,2), ...}
    Reduce 2-class logits to a single positive-class logit via
    ``logits[:,1] - logits[:,0]``.

Output JSONL is the same schema as ``scripts/score_tiles.py``:
``{tile_id, label, prob, logit, split}``.

Usage::

    python scripts/score_tiles_codesign.py \\
        --variant shared --backbone resnet18 \\
        --weights runs/codesign_ships_shared_resnet18/best.pt \\
        --data-root data/processed/dota_ships --split val \\
        --device 1 --out reports/gate_scores/codesign_ships_shared_resnet18_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.codesign import (  # noqa: E402
    EarlyExitWrapper,
    RelaxedGatingCascade,
    RelaxedGatingCascadeConfig,
    SharedBackboneCascade,
)
from src.gate import GateDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--variant", choices=("shared", "early_exit", "relaxed"), required=True)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--weights", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--exit-at", type=int, default=1, help="early_exit variant only")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _build(args: argparse.Namespace) -> torch.nn.Module:
    if args.variant == "shared":
        return SharedBackboneCascade(backbone_name=args.backbone, num_classes=1)
    if args.variant == "early_exit":
        return EarlyExitWrapper(backbone_name=args.backbone, exit_at=args.exit_at, num_classes=1)
    if args.variant == "relaxed":
        return RelaxedGatingCascade(RelaxedGatingCascadeConfig(backbone=args.backbone))
    raise ValueError(args.variant)


def _gate_logit(pred: dict, variant: str) -> torch.Tensor:
    if variant == "relaxed":
        # Reduce {drop, keep} 2-class logits into a single positive logit.
        gl = pred["gate_logits"]  # (B, 2)
        return gl[:, 1] - gl[:, 0]
    return pred["gate_logit"]


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    model = _build(args).to(device)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = state.get("model") or state.get("model_state_dict") or state
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = GateDataset(root=args.data_root, split=args.split, image_size=args.image_size, augment=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as h, torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            pred = model(images)
            logits = _gate_logit(pred, args.variant).float().cpu()
            probs = torch.sigmoid(logits).numpy()
            logits_np = logits.numpy()
            for i, tid in enumerate(batch["tile_id"]):
                h.write(json.dumps({
                    "tile_id": tid,
                    "label": int(batch["label"][i]),
                    "logit": float(logits_np[i]),
                    "prob": float(probs[i]),
                    "split": args.split,
                }) + "\n")
                n += 1
    print(f"[score-codesign] wrote {n} scored tiles to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
