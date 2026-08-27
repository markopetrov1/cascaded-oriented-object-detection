#!/usr/bin/env python3
"""Train a gate via distillation from detector outputs (Pillar 3, variant 3).

The detector is run once over every tile (e.g. via ``ultralytics yolo predict``)
and the per-tile max-objectness becomes the soft target. We then train any
gate backbone with MSE on the soft target — optionally mixed with the binary
hard label.

Usage::

    python scripts/train_distill.py \\
        --gate-config configs/experiments/gate_resnet18.yaml \\
        --detector-runs runs/baseline_yolo11m_dota_ships_obb/predict_train \\
        --metadata-jsonl data/processed/dota_ships/metadata/tiles.jsonl \\
        --epochs 15 --device 0 --name gate_resnet18_distill
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

from src.codesign import DistillationConfig, distillation_loss, soft_target_from_detector  # noqa: E402
from src.gate import GateDataset, build_gate_model, make_balanced_sampler  # noqa: E402
from src.utils.paths import read_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--gate-config", required=True, help="Reuse a normal gate config for hyperparams")
    p.add_argument("--detector-runs", required=True, help="Ultralytics predict output dir; expects labels/<tile>.txt with scores")
    p.add_argument("--metadata-jsonl", required=True, help="prepare_dota tiles.jsonl, used for hard labels")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--soft-weight", type=float, default=1.0)
    p.add_argument("--hard-weight", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return p.parse_args()


def _load_detector_scores(runs_dir: Path) -> dict[str, np.ndarray]:
    labels_dir = runs_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Expected ultralytics predict labels at {labels_dir}")
    out: dict[str, np.ndarray] = {}
    for txt in sorted(labels_dir.glob("*.txt")):
        scores: list[float] = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 10:  # cls + 8 coords + score
                scores.append(float(parts[9]))
        out[txt.stem] = np.asarray(scores, dtype=np.float32) if scores else np.zeros(0, dtype=np.float32)
    return out


def main() -> int:
    args = parse_args()
    cfg = read_yaml(args.gate_config)
    device = (
        f"cuda:{args.device}" if (args.device and args.device.isdigit())
        else (args.device or cfg.get("device", "cuda:0"))
    )
    epochs = args.epochs or cfg.get("epochs", 20)
    name = args.name or f"{cfg.get('name', 'gate')}_distill"
    run_dir = Path(cfg.get("project", "runs")) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = GateDataset(cfg["data_root"], cfg["train_split"], image_size=cfg.get("image_size", 256), augment=True)
    val_ds = GateDataset(cfg["data_root"], cfg["val_split"], image_size=cfg.get("image_size", 256), augment=False)

    detector_scores = _load_detector_scores(Path(args.detector_runs))
    soft_targets_train = soft_target_from_detector(detector_scores, [tid for _, _, tid in zip(train_ds.items, train_ds.items, [it[0].stem for it in train_ds.items])])
    # The triple-zip above is awkward; collapse to a clean loop:
    soft_targets_train = soft_target_from_detector(detector_scores, [it[0].stem for it in train_ds.items])

    sampler = make_balanced_sampler(train_ds.labels, cfg.get("pos_neg_ratio", 1.0 / 3.0)) if cfg.get("sampler") == "balanced" else None
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 64), sampler=sampler, shuffle=sampler is None, num_workers=cfg.get("num_workers", 8), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=cfg.get("num_workers", 8), pin_memory=True)

    model = build_gate_model(cfg["backbone"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    distill_cfg = DistillationConfig(soft_target_weight=args.soft_weight, hard_target_weight=args.hard_weight, temperature=args.temperature)

    tile_id_to_soft = {it[0].stem: float(soft_targets_train[i]) for i, it in enumerate(train_ds.items)}
    history = []
    best_val_pr_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True) and "cuda" in str(device))

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(-1)
            soft = torch.tensor(
                [tile_id_to_soft.get(t, 0.0) for t in batch["tile_id"]],
                dtype=torch.float32, device=device,
            ).unsqueeze(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.get("amp", True) and "cuda" in str(device)):
                logits = model(images)
                loss = distillation_loss(logits, soft, labels if args.hard_weight > 0 else None, distill_cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * images.size(0)
            n += images.size(0)
        scheduler.step()

        # Quick val PR-AUC for monitoring (binary head against hard labels).
        model.eval()
        from sklearn.metrics import average_precision_score

        val_probs: list[float] = []
        val_labels: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=cfg.get("amp", True) and "cuda" in str(device)):
                    logits = model(images).squeeze(-1)
                val_probs.extend(torch.sigmoid(logits.float()).detach().cpu().numpy().tolist())
                val_labels.extend(batch["label"].numpy().tolist())
        pr_auc = float(average_precision_score(val_labels, val_probs)) if any(val_labels) and not all(val_labels) else float("nan")
        print(f"[distill] epoch {epoch}/{epochs} train_loss={total / max(n, 1):.4f} val_pr_auc={pr_auc:.4f}")
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_pr_auc": pr_auc})
        if pr_auc > best_val_pr_auc:
            best_val_pr_auc = pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "config": cfg, "backbone": cfg["backbone"]}, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[distill] early stopping at epoch {epoch} (no improvement for {args.patience} epochs; best={best_val_pr_auc:.4f} @ epoch {best_epoch})")
                break

    torch.save({"model": model.state_dict(), "config": cfg, "backbone": cfg["backbone"]}, run_dir / "last.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[distill] done. best val PR-AUC = {best_val_pr_auc:.4f}; run_dir = {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
