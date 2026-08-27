"""Stage-1 gate: a lightweight binary classifier that decides whether a tile
contains any object of interest. Reads tile images from
``data/processed/<run>/images/{split}/`` and binary labels from
``data/processed/<run>/gate_labels/{split}/``.

Critical design points:
- The gate's success metric is *not* accuracy or F1 — it is downstream-mAP at a
  given total compute. We train with BCE/focal here for stability, and let
  Phase-2 calibration + Phase-3 cascade evaluation pick the threshold that
  optimizes mAP-at-compute. See research plan §7.2 / §15.7.
- Negatives outnumber positives by ~5-20x in DOTA-Ships, so we expose three
  imbalance strategies: BCE, focal, and ratio-controlled batches. Pick one per
  config and report all three for the strongest backbone.
- AMP is FP16 (RTX 8000 Turing — no BF16). Loss-scaling handled by
  ``torch.cuda.amp``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------- Dataset --------------------------------------------------------


class GateDataset(Dataset):
    """Reads tile images + per-tile binary labels.

    Layout convention (matches :func:`src.datasets.dota.prepare_dota`):
        <root>/images/<split>/<tile_id>.<ext>
        <root>/gate_labels/<split>/<tile_id>.txt   # contains '0' or '1'
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        augment: bool = False,
        tile_ids: list[str] | None = None,
        image_suffix: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        images_dir = self.root / "images" / split
        gate_dir = self.root / "gate_labels" / split
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {images_dir}")
        if not gate_dir.exists():
            raise FileNotFoundError(f"Missing gate_labels dir: {gate_dir}")

        suffix = image_suffix or self._detect_suffix(images_dir)
        all_paths = sorted(images_dir.glob(f"*{suffix}"))
        if tile_ids is not None:
            wanted = set(tile_ids)
            all_paths = [p for p in all_paths if p.stem in wanted]
        self.items: list[tuple[Path, int]] = []
        for img_path in all_paths:
            label_path = gate_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            label = int(label_path.read_text(encoding="utf-8").strip() or "0")
            self.items.append((img_path, label))
        if not self.items:
            raise RuntimeError(f"No (image, label) pairs found under {self.root} split={split}")

        self._labels = np.asarray([lbl for _, lbl in self.items], dtype=np.int64)

        # Build the augmentation pipeline lazily — albumentations is optional at
        # import time so smoke imports work without the install.
        self._transform = None
        if augment:
            self._transform = self._build_train_transform()
        else:
            self._transform = self._build_eval_transform()

    @staticmethod
    def _detect_suffix(images_dir: Path) -> str:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            if next(images_dir.glob(f"*{ext}"), None) is not None:
                return ext
        return ".png"

    def _build_train_transform(self):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose(
            [
                A.LongestMaxSize(self.image_size),
                A.PadIfNeeded(self.image_size, self.image_size, border_mode=0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3, brightness_limit=0.15, contrast_limit=0.15),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def _build_eval_transform(self):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose(
            [
                A.LongestMaxSize(self.image_size),
                A.PadIfNeeded(self.image_size, self.image_size, border_mode=0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_path, label = self.items[index]
        import cv2

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read tile image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self._transform(image=img)
        return {
            "image": out["image"],
            "label": torch.tensor(label, dtype=torch.float32),
            "tile_id": img_path.stem,
        }


def make_balanced_sampler(labels: np.ndarray, pos_neg_ratio: float = 1.0 / 3.0) -> WeightedRandomSampler:
    """Sample so the expected positive fraction in a batch is ``pos_neg_ratio``.

    Default 1:3 (pos:neg) matches the research plan §11 recommendation and
    keeps gradient updates from being dominated by negatives. The number of
    samples per epoch is held to ``len(labels)`` so epoch-time stays comparable
    across imbalance strategies.
    """
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        # Degenerate; fall back to uniform sampling.
        return WeightedRandomSampler(weights=[1.0] * n, num_samples=n, replacement=True)
    pos_w = pos_neg_ratio / max(n_pos, 1)
    neg_w = (1.0 - pos_neg_ratio) / max(n_neg, 1)
    weights = np.where(labels == 1, pos_w, neg_w)
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=n, replacement=True)


# ---------- Models ---------------------------------------------------------


def build_gate_model(name: str, pretrained: bool = True) -> nn.Module:
    """Return a binary classifier with a single-logit head.

    Names: ``resnet18``, ``resnet50``, ``mobilenetv3_small_100``,
    ``mobilenetv3_large_100``, ``efficientnet_b0``, ``tiny`` (custom ~1M-param).
    """
    if name == "tiny":
        return _build_tiny()
    import timm

    model = timm.create_model(name, pretrained=pretrained, num_classes=1)
    return model


def _build_tiny() -> nn.Module:
    """Custom 3-block ConvNet; ~1M params with 256-px input. Hand-rolled so we
    have a backbone with no ImageNet bias as a sanity-check baseline.
    """

    def block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(3, 32),
        block(32, 64),
        block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 1),
    )


# ---------- Losses ---------------------------------------------------------


class FocalBCE(nn.Module):
    """Binary focal loss. ``alpha`` weights the positive class; ``gamma`` is the
    standard focusing parameter. For RS gating with ~10-20% positive rate,
    ``alpha=0.75 gamma=2.0`` is a reasonable starting point.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()


def build_loss(kind: str, **kwargs: Any) -> nn.Module:
    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    if kind == "weighted_bce":
        pos_weight = kwargs.get("pos_weight")
        if pos_weight is None:
            raise ValueError("weighted_bce requires pos_weight")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
    if kind == "focal":
        return FocalBCE(alpha=kwargs.get("alpha", 0.75), gamma=kwargs.get("gamma", 2.0))
    raise ValueError(f"Unknown loss kind: {kind}")


# ---------- Metrics --------------------------------------------------------


def _binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (probs >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
    }


def _pr_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(average_precision_score(labels, probs))


# ---------- Training -------------------------------------------------------


@dataclass
class GateTrainConfig:
    backbone: str
    image_size: int = 256
    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "bce"  # bce | weighted_bce | focal
    pos_weight: float | None = None
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    sampler: str = "uniform"  # uniform | balanced
    pos_neg_ratio: float = 1.0 / 3.0
    amp: bool = True
    device: str = "cuda:0"
    seed: int = 67
    name: str = "gate_run"
    project: str = "runs"
    data_root: str = "data/processed/dota_ships"
    train_split: str = "train"
    val_split: str = "val"
    train_tile_ids: list[str] | None = None
    val_tile_ids: list[str] | None = None
    log_every: int = 50
    use_wandb: bool = False
    wandb_project: str = "cascaded-rs-detection"
    patience: int = 10  # epochs of no PR-AUC improvement before early stop


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(spec).isdigit():
        return torch.device(f"cuda:{spec}")
    return torch.device(spec)


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_gate(cfg: GateTrainConfig) -> dict[str, Any]:
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    run_dir = Path(cfg.project) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = GateDataset(
        cfg.data_root, cfg.train_split, image_size=cfg.image_size, augment=True,
        tile_ids=cfg.train_tile_ids,
    )
    val_ds = GateDataset(
        cfg.data_root, cfg.val_split, image_size=cfg.image_size, augment=False,
        tile_ids=cfg.val_tile_ids,
    )
    if cfg.sampler == "balanced":
        sampler = make_balanced_sampler(train_ds.labels, cfg.pos_neg_ratio)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = build_gate_model(cfg.backbone).to(device)
    loss_kwargs: dict[str, Any] = {"alpha": cfg.focal_alpha, "gamma": cfg.focal_gamma}
    if cfg.loss == "weighted_bce" and cfg.pos_weight is None:
        # Auto-compute pos_weight as n_neg / n_pos.
        pos = max(int(train_ds.labels.sum()), 1)
        neg = max(int((1 - train_ds.labels).sum()), 1)
        loss_kwargs["pos_weight"] = neg / pos
    elif cfg.pos_weight is not None:
        loss_kwargs["pos_weight"] = cfg.pos_weight
    criterion = build_loss(cfg.loss, **loss_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.name, config=cfg.__dict__)
        except Exception as exc:
            print(f"[gate] wandb init failed; continuing without: {exc}")

    history: list[dict[str, Any]] = []
    best_pr_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch
        )
        val_metrics = _evaluate(model, val_loader, device, cfg.amp)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[gate] epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"val_recall@0.5={val_metrics['recall']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(row)

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {"model": model.state_dict(), "config": cfg.__dict__, "val_metrics": val_metrics},
                run_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(
                    f"[gate] early stopping at epoch {epoch} "
                    f"(no improvement for {cfg.patience} epochs; best={best_pr_auc:.4f} @ epoch {best_epoch})"
                )
                break

    torch.save(
        {"model": model.state_dict(), "config": cfg.__dict__},
        run_dir / "last.pt",
    )
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    if wandb_run is not None:
        wandb_run.finish()
    return {"history": history, "best_pr_auc": best_pr_auc, "run_dir": str(run_dir)}


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: GateTrainConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach()) * images.size(0)
        n += images.size(0)
        if step % cfg.log_every == 0:
            print(
                f"[gate] epoch {epoch} step {step}/{len(loader)} "
                f"loss={total_loss / max(n, 1):.4f}"
            )
    elapsed = time.time() - start
    return {"loss": total_loss / max(n, 1), "elapsed_sec": elapsed}


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, float]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(images).squeeze(-1)
        probs.extend(torch.sigmoid(logits.float()).detach().cpu().numpy().tolist())
        labels.extend(batch["label"].numpy().tolist())
    p = np.asarray(probs)
    y = np.asarray(labels, dtype=np.int64)
    metrics = _binary_metrics(p, y, threshold=0.5)
    metrics["pr_auc"] = _pr_auc(p, y)
    metrics["positive_rate"] = float(y.mean())
    return metrics


# ---------- Inference / scoring -------------------------------------------


@torch.no_grad()
def score_tiles(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    out_jsonl: str | Path,
    image_size: int = 256,
    batch_size: int = 128,
    num_workers: int = 8,
    device: str = "auto",
    amp: bool = True,
    backbone: str | None = None,
) -> Path:
    """Run a trained gate over a split and emit per-tile (logit, prob) JSONL.

    The output is consumed by every downstream cascade evaluator (calibration,
    Pareto sweep, oracle comparison). One row per tile::

        {"tile_id": "P0003__x0_y0_w1024_h1024", "split": "val",
         "label": 0, "logit": -3.21, "prob": 0.0386}
    """
    dev = _resolve_device(device if device != "auto" else "auto")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    backbone = backbone or cfg_dict.get("backbone")
    if not backbone:
        raise ValueError("Could not infer backbone; pass backbone= explicitly")
    model = build_gate_model(backbone, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    ds = GateDataset(data_root, split, image_size=image_size, augment=False)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for batch in loader:
            images = batch["image"].to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and dev.type == "cuda"):
                logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
            logits_np = logits.float().detach().cpu().numpy()
            for i, tile_id in enumerate(batch["tile_id"]):
                row = {
                    "tile_id": tile_id,
                    "split": split,
                    "label": int(batch["label"][i].item()),
                    "logit": float(logits_np[i]),
                    "prob": float(probs[i]),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
    print(f"[gate] wrote {written} scored tiles to {out_path}")
    return out_path


def load_score_jsonl(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Helper: read a score JSONL into ``(probs, labels, tile_ids)``."""
    probs: list[float] = []
    labels: list[int] = []
    ids: list[str] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        probs.append(float(row["prob"]))
        labels.append(int(row["label"]))
        ids.append(str(row["tile_id"]))
    return np.asarray(probs), np.asarray(labels, dtype=np.int64), ids


def gate_filter_curve(
    probs: np.ndarray, labels: np.ndarray, n_thresholds: int = 101
) -> list[dict[str, float]]:
    """Sweep thresholds and return (threshold, recall, filter_rate) triples.
    The ``filter_rate`` is the fraction of tiles the gate would *reject* —
    i.e. ``mean(prob < threshold)``. This is the curve every cascade Pareto plot
    consumes.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows: list[dict[str, float]] = []
    n_pos = max(int(labels.sum()), 1)
    for t in thresholds:
        kept = probs >= t
        tp = int(((kept) & (labels == 1)).sum())
        rec = tp / n_pos
        rows.append(
            {"threshold": float(t), "recall": float(rec), "filter_rate": float(1.0 - kept.mean())}
        )
    return rows


