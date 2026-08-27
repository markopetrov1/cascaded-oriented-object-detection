#!/usr/bin/env python3
"""Train a co-design cascade variant — shared-backbone, early-exit, or
relaxed-gating.

Each variant trains a single ``nn.Module`` from ``src.codesign`` against the
prepared DOTA tiles. The OBB head used here is the in-house
``SimpleOBBHead``, which is intentionally simpler than YOLO's tuned head — it
exists so we can ablate co-design *architecture* effects without confounding
them with detector-specific tuning. Reported numbers should be compared
against an *independent* baseline trained with the same head.

Usage::

    python scripts/train_codesign.py \\
        --variant shared --backbone resnet18 \\
        --data-root data/processed/dota_ships --splits-yaml configs/splits/geographic_v1.yaml \\
        --epochs 30 --device 0 --name codesign_shared_resnet18

Variants:
  - ``shared``: SharedBackboneCascade
  - ``early_exit``: EarlyExitWrapper
  - ``relaxed``: RelaxedGatingCascade (Gumbel-softmax, end-to-end)

Note: the OBB target builder (``_build_dense_targets``) emits dense per-cell
ground truth from the YOLO-OBB tile labels. Cell stride is
``image_size / output_feature_size``; the head outputs are at the last
backbone feature map's resolution (typically /32). We treat each positive
cell as a positive anchor at scale=1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.codesign import (  # noqa: E402
    EarlyExitWrapper,
    RelaxedGatingCascade,
    RelaxedGatingCascadeConfig,
    SharedBackboneCascade,
    relaxed_gating_loss,
    shared_backbone_loss,
)
from src.gate import GateDataset, make_balanced_sampler  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--variant", choices=("shared", "early_exit", "relaxed"), required=True)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gate-weight", type=float, default=1.0)
    p.add_argument("--obb-weight", type=float, default=0.1,
                   help="OBB loss weight. Default 0.1 (was 1.0): the dense per-cell BCE in SimpleOBBHead "
                        "dominates gradient flow when set to 1.0 and corrupts backbone features that the "
                        "gate task needs, producing the 'best val PR-AUC at epoch 1, then decline' pathology.")
    p.add_argument("--gate-warmup-epochs", type=int, default=5,
                   help="For the first N epochs, set obb_weight=0 so the backbone learns gate-friendly "
                        "features without OBB-loss interference. After warmup, obb_weight ramps to its "
                        "configured value over the next 3 epochs.")
    p.add_argument("--drop-reward", type=float, default=0.05, help="relaxed variant only")
    p.add_argument("--exit-at", type=int, default=1, help="early_exit variant only — feature index for the gate head")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--name", required=True)
    p.add_argument("--project", default="runs")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without val PR-AUC improvement)")
    return p.parse_args()


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.variant == "shared":
        return SharedBackboneCascade(backbone_name=args.backbone, num_classes=1)
    if args.variant == "early_exit":
        return EarlyExitWrapper(backbone_name=args.backbone, exit_at=args.exit_at, num_classes=1)
    if args.variant == "relaxed":
        return RelaxedGatingCascade(
            RelaxedGatingCascadeConfig(backbone=args.backbone)
        )
    raise ValueError(args.variant)


def _build_dense_targets(
    tile_labels_path: Path, image_size: int, feat_h: int, feat_w: int, num_classes: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a YOLO-OBB tile label file into dense (cls, box, angle, mask)
    targets at the head's feature resolution.
    """
    cls = torch.zeros((num_classes, feat_h, feat_w), dtype=torch.float32)
    box = torch.zeros((4, feat_h, feat_w), dtype=torch.float32)
    angle = torch.zeros((2, feat_h, feat_w), dtype=torch.float32)
    mask = torch.zeros((1, feat_h, feat_w), dtype=torch.bool)
    if not tile_labels_path.exists():
        return cls, box, angle, mask
    for line in tile_labels_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        c = int(parts[0])
        coords = np.asarray([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2) * image_size
        cx, cy = float(coords[:, 0].mean()), float(coords[:, 1].mean())
        w = float(np.linalg.norm(coords[1] - coords[0]))
        h = float(np.linalg.norm(coords[2] - coords[1]))
        theta = float(np.arctan2(coords[1, 1] - coords[0, 1], coords[1, 0] - coords[0, 0]))
        gx = int(min(max(cx * feat_w / image_size, 0), feat_w - 1))
        gy = int(min(max(cy * feat_h / image_size, 0), feat_h - 1))
        cls[c, gy, gx] = 1.0
        box[0, gy, gx] = cx / image_size
        box[1, gy, gx] = cy / image_size
        box[2, gy, gx] = w / image_size
        box[3, gy, gx] = h / image_size
        angle[0, gy, gx] = float(np.sin(theta))
        angle[1, gy, gx] = float(np.cos(theta))
        mask[0, gy, gx] = True
    return cls, box, angle, mask


class CodesignDataset(GateDataset):
    """Extends ``GateDataset`` with on-the-fly OBB dense targets."""

    def __init__(self, root, split, image_size=256, augment=False, feat_size=8, **kwargs):
        super().__init__(root=root, split=split, image_size=image_size, augment=augment, **kwargs)
        self.feat_size = feat_size
        self.label_dir = Path(root) / "labels" / split

    def __getitem__(self, index):
        item = super().__getitem__(index)
        tile_id = item["tile_id"]
        cls, box, angle, mask = _build_dense_targets(
            self.label_dir / f"{tile_id}.txt", self.image_size, self.feat_size, self.feat_size
        )
        item["obb_cls"] = cls
        item["obb_box"] = box
        item["obb_angle"] = angle
        item["obb_mask"] = mask
        return item


def _resolve_feat_size(model: torch.nn.Module, image_size: int) -> int:
    """Probe the backbone with a dummy input to learn the last feature map's
    spatial resolution. Used by CodesignDataset to build matching dense targets.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size, device=device)
        if hasattr(model, "backbone"):
            feats = model.backbone(x)
            return int(feats[-1].shape[-1])
    return image_size // 32


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = _build_model(args).to(args.device)
    feat_size = _resolve_feat_size(model, args.image_size)
    print(f"[codesign] variant={args.variant} backbone={args.backbone} feat_size={feat_size}")

    train_ds = CodesignDataset(args.data_root, args.train_split, image_size=args.image_size, augment=True, feat_size=feat_size)
    val_ds = CodesignDataset(args.data_root, args.val_split, image_size=args.image_size, augment=False, feat_size=feat_size)
    sampler = make_balanced_sampler(train_ds.labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and "cuda" in args.device)

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val_pr_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # Gate-task-first schedule: zero OBB weight during warmup, ramp linearly over 3 epochs.
        if epoch <= args.gate_warmup_epochs:
            current_obb_weight = 0.0
        elif epoch <= args.gate_warmup_epochs + 3:
            ramp = (epoch - args.gate_warmup_epochs) / 3.0
            current_obb_weight = args.obb_weight * ramp
        else:
            current_obb_weight = args.obb_weight
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            images = batch["image"].to(args.device, non_blocking=True)
            gate_target = batch["label"].to(args.device, non_blocking=True)
            obb_targets = {
                "cls": batch["obb_cls"].to(args.device, non_blocking=True),
                "box": batch["obb_box"].to(args.device, non_blocking=True),
                "angle": batch["obb_angle"].to(args.device, non_blocking=True),
            }
            obb_mask = batch["obb_mask"].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and "cuda" in args.device):
                pred = model(images)
                if args.variant == "relaxed":
                    losses = relaxed_gating_loss(pred, obb_targets, obb_mask, model.cfg)
                else:
                    losses = shared_backbone_loss(pred, gate_target, obb_targets, obb_mask, gate_weight=args.gate_weight, obb_weight=current_obb_weight)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(losses["total"].detach()) * images.size(0)
            n += images.size(0)
        scheduler.step()

        # Cheap monitoring: gate's PR-AUC on val.
        model.eval()
        from sklearn.metrics import average_precision_score

        probs: list[float] = []
        labels: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(args.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp and "cuda" in args.device):
                    pred = model(images)
                if "gate_logit" in pred:
                    p = torch.sigmoid(pred["gate_logit"].float()).detach().cpu().numpy().tolist()
                elif "keep_mask" in pred:
                    p = pred["keep_mask"].float().detach().cpu().numpy().tolist()
                else:
                    p = [0.5] * images.size(0)
                probs.extend(p)
                labels.extend(batch["label"].numpy().tolist())
        if any(labels) and not all(labels):
            pr_auc = float(average_precision_score(labels, probs))
        else:
            pr_auc = float("nan")
        print(f"[codesign] epoch {epoch}/{args.epochs} train_loss={total / max(n, 1):.4f} val_pr_auc={pr_auc:.4f} obb_w={current_obb_weight:.3f}")
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_pr_auc": pr_auc})
        if pr_auc > best_val_pr_auc:
            best_val_pr_auc = pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "feat_size": feat_size}, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[codesign] early stopping at epoch {epoch} (no improvement for {args.patience} epochs; best={best_val_pr_auc:.4f} @ epoch {best_epoch})")
                break

    torch.save({"model": model.state_dict(), "args": vars(args), "feat_size": feat_size}, run_dir / "last.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[codesign] done. best val PR-AUC = {best_val_pr_auc:.4f}; run_dir = {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
