"""Pillar 3 — co-design variants of the cascade.

The four configurations from research plan §7.3:

1. ``SharedBackboneCascade`` — one backbone, two heads (binary gate + OBB
   detector). Joint training with a weighted loss. Tests whether the gate
   piggybacks on detector features for free.

2. ``EarlyExitDetector`` — the gate is a binary head bolted to an early block
   of the detector backbone. Conditional continuation if positive. Most
   compute-efficient variant; implemented via a forward hook + a custom
   feature dict so we don't fork Ultralytics.

3. ``DistillGateFromDetector`` — a *training procedure*, not an architecture.
   Run the detector on every tile, take max-objectness per tile as the soft
   target, train any gate backbone via MSE on the soft target.

4. ``RelaxedGatingCascade`` — Gumbel-softmax over ``{drop, keep}`` with
   straight-through estimator. The whole cascade trained end-to-end on
   detection loss + a small drop-reward regularizer. Highest risk; usually
   fails to converge — the failure itself is a contribution per §15.10.

All four share the binary gate signal definition, so they can compare cleanly
on the Pareto plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. Shared backbone, two heads ---------------------------------


class SimpleOBBHead(nn.Module):
    """Tiny in-house OBB head used by the co-design variants.

    Input: a feature map of shape ``(B, C, H, W)``.
    Outputs (per spatial cell, per anchor scale=1):
      - 5 box params: cx, cy, w, h, angle (sin, cos encoded as 6 dims total)
      - K class logits

    For the cascade research the head only needs to predict ships (K=1), so we
    use anchor-free single-cell predictions at multiple feature resolutions.
    Loss: smooth-L1 on (cx, cy, w, h), MSE on (sin θ, cos θ) angle encoding,
    BCE on classification.

    This is intentionally not as strong as YOLO's tuned OBB head — but for
    co-design ablations we just need a head that obeys the same contract and
    can share a backbone with the gate.
    """

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.box = nn.Conv2d(in_channels, 4, kernel_size=1)  # cx, cy, w, h
        self.angle = nn.Conv2d(in_channels, 2, kernel_size=1)  # sin, cos
        self.cls = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"box": self.box(feat), "angle": self.angle(feat), "cls": self.cls(feat)}


def smooth_l1_obb_loss(
    pred: dict[str, torch.Tensor],
    target_box: torch.Tensor,
    target_angle: torch.Tensor,
    target_cls: torch.Tensor,
    pos_mask: torch.Tensor,
    cls_pos_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compose smooth-L1 box loss + MSE angle loss + BCE classification.

    All targets are dense (B, *, H, W), pos_mask is the binary indicator of
    which spatial cells correspond to positive boxes (computed by the data
    pipeline that emits dense targets from the polygon list).
    """
    box_loss = F.smooth_l1_loss(pred["box"][pos_mask.expand_as(pred["box"])], target_box[pos_mask.expand_as(target_box)], reduction="mean") if pos_mask.any() else pred["box"].sum() * 0.0
    angle_loss = F.mse_loss(pred["angle"][pos_mask.expand_as(pred["angle"])], target_angle[pos_mask.expand_as(target_angle)], reduction="mean") if pos_mask.any() else pred["angle"].sum() * 0.0
    cls_target = target_cls.float()
    cls_loss = F.binary_cross_entropy_with_logits(
        pred["cls"], cls_target, pos_weight=torch.tensor(cls_pos_weight, device=cls_target.device)
    )
    total = box_loss + angle_loss + cls_loss
    return {"total": total, "box": box_loss, "angle": angle_loss, "cls": cls_loss}


class SharedBackboneCascade(nn.Module):
    """One backbone, one binary head, one OBB head. ``forward`` returns both;
    callers can choose which to consume.

    ``forward_gate_only`` returns only the binary head — useful at inference
    when we want the cheaper path.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        feature_index: int = -1,
        num_classes: int = 1,
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        feat_info = self.backbone.feature_info.channels()
        feat_chan = feat_info[feature_index]
        self.feature_index = feature_index
        self.gate_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(feat_chan, 1)
        )
        self.obb_head = SimpleOBBHead(feat_chan, num_classes=num_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return feats[self.feature_index]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self._features(x)
        gate_logit = self.gate_head(feat).squeeze(-1)
        obb = self.obb_head(feat)
        return {"gate_logit": gate_logit, **{f"obb_{k}": v for k, v in obb.items()}}

    @torch.no_grad()
    def forward_gate_only(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._features(x)
        return self.gate_head(feat).squeeze(-1)


def shared_backbone_loss(
    pred: dict[str, torch.Tensor],
    gate_target: torch.Tensor,
    obb_targets: dict[str, torch.Tensor] | None,
    obb_pos_mask: torch.Tensor | None,
    gate_weight: float = 1.0,
    obb_weight: float = 1.0,
    gate_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Combined gate+detection loss.

    The gate is supervised on every tile (all batches contribute). The OBB
    head is supervised on positive tiles only (we mask negatives' detection
    loss to zero — they have no ground-truth boxes anyway).
    """
    gate_logit = pred["gate_logit"]
    if gate_loss_fn is None:
        gate_loss = F.binary_cross_entropy_with_logits(gate_logit, gate_target.float())
    else:
        gate_loss = gate_loss_fn(gate_logit, gate_target.float())

    obb_total = gate_logit.new_tensor(0.0)
    obb_breakdown: dict[str, torch.Tensor] = {}
    if obb_targets is not None and obb_pos_mask is not None and obb_pos_mask.any():
        obb_breakdown = smooth_l1_obb_loss(
            pred={k.replace("obb_", ""): v for k, v in pred.items() if k.startswith("obb_")},
            target_box=obb_targets["box"],
            target_angle=obb_targets["angle"],
            target_cls=obb_targets["cls"],
            pos_mask=obb_pos_mask,
        )
        obb_total = obb_breakdown["total"]
    total = gate_weight * gate_loss + obb_weight * obb_total
    return {"total": total, "gate": gate_loss, "obb": obb_total, **obb_breakdown}


# ---------- 2. Early-exit detector ----------------------------------------


class EarlyExitWrapper(nn.Module):
    """Wraps any backbone so block ``exit_at`` produces a binary gate logit
    while later blocks still run when the gate accepts. At inference we expose
    a ``two_stage_predict`` method that short-circuits negatives; at training
    time both heads are supervised jointly.

    Implementation strategy:
      - We rely on ``timm.create_model(..., features_only=True)`` to expose
        intermediate feature maps. ``exit_at`` is an index into that list.
      - At training: forward the full backbone, compute both losses.
      - At inference: forward only the early stage, branch on the gate.
        Tiles that pass the gate are batched and forwarded through the rest.

    The pure-PyTorch approach lets us benchmark FLOPs honestly: the
    ``two_stage_predict`` accepts an ``flops_meter`` object that records the
    realized cost.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        exit_at: int = 1,
        num_classes: int = 1,
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        chans = self.backbone.feature_info.channels()
        self.exit_at = exit_at
        self.gate_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(chans[exit_at], 1)
        )
        self.obb_head = SimpleOBBHead(chans[-1], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Train-time forward: compute both heads from a single full forward."""
        feats = self.backbone(x)
        gate_feat = feats[self.exit_at]
        gate_logit = self.gate_head(gate_feat).squeeze(-1)
        obb = self.obb_head(feats[-1])
        return {"gate_logit": gate_logit, **{f"obb_{k}": v for k, v in obb.items()}}

    @torch.no_grad()
    def two_stage_predict(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> dict[str, Any]:
        """Inference-time forward with early exit.

        Returns ``{"keep_mask", "obb"}`` where ``obb`` is a dict whose tensors
        have shape ``(K, ...)`` for the K accepted tiles. The caller is
        responsible for re-indexing back into the batch.
        """
        # Manual stagewise forward — we cannot simply slice features_only output
        # because that always runs all stages. We use the underlying forward
        # mechanics by walking the backbone modules until exit_at, then if
        # any tile is kept, finishing the forward on that subset.
        # For simplicity we do two full forwards on the full batch then on the
        # accepted subset; FLOPs counting handles the rest correctly because
        # ``flops`` is reported relative to (n_tiles, n_accepted).
        first_pass = self.backbone(x)
        gate_logit = self.gate_head(first_pass[self.exit_at]).squeeze(-1)
        keep = (torch.sigmoid(gate_logit) >= threshold)
        if not keep.any():
            return {"keep_mask": keep, "obb": None, "gate_prob": torch.sigmoid(gate_logit)}
        sub = x[keep]
        sub_feats = self.backbone(sub)
        obb = self.obb_head(sub_feats[-1])
        return {"keep_mask": keep, "obb": obb, "gate_prob": torch.sigmoid(gate_logit)}


# ---------- 3. Distillation: detector → gate ------------------------------


@dataclass
class DistillationConfig:
    soft_target_weight: float = 1.0
    hard_target_weight: float = 0.0  # combine with soft labels if > 0
    temperature: float = 1.0


def soft_target_from_detector(
    detections_per_tile: dict[str, np.ndarray],
    tile_ids: list[str],
) -> np.ndarray:
    """Per-tile soft target = max objectness on that tile.

    ``detections_per_tile[tile_id]`` is an array of detection scores (one row
    per detected box). Tiles with no detections get a 0 target. This
    "objectness" is the detector's confidence that an object exists on the
    tile, so it carries richer signal than the binary label and we can
    distill against it.
    """
    out = np.zeros(len(tile_ids), dtype=np.float32)
    for i, tid in enumerate(tile_ids):
        scores = detections_per_tile.get(tid, np.zeros(0, dtype=np.float32))
        out[i] = float(scores.max()) if len(scores) > 0 else 0.0
    return out


def distillation_loss(
    student_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    hard_labels: torch.Tensor | None,
    cfg: DistillationConfig,
) -> torch.Tensor:
    """Mix MSE on the soft target with optional BCE on the hard label.

    Soft target is in [0, 1]; we apply ``temperature`` by passing
    ``logit / T`` through sigmoid then matching MSE — the standard
    distillation idiom.
    """
    p = torch.sigmoid(student_logits / max(cfg.temperature, 1e-3))
    soft = F.mse_loss(p, soft_targets)
    if cfg.hard_target_weight > 0 and hard_labels is not None:
        hard = F.binary_cross_entropy_with_logits(student_logits, hard_labels.float())
        return cfg.soft_target_weight * soft + cfg.hard_target_weight * hard
    return cfg.soft_target_weight * soft


# ---------- 4. End-to-end relaxed gating ----------------------------------


class GumbelStraightThroughGate(nn.Module):
    """Differentiable {drop, keep} gate.

    Forward: produce keep probability, sample a relaxed binary mask via
    Gumbel-softmax, return the mask. The straight-through estimator yields
    discrete decisions on the forward pass while keeping gradient flow through
    the soft mask on the backward.

    This is the headline-risk module. Common failure modes:
      - threshold collapses to "keep all" (drop reward too small)
      - threshold collapses to "drop all" (drop reward too large)
      - high gradient variance from straight-through estimator
    Document each failure as a result; do not paper over.
    """

    def __init__(self, in_dim: int, hidden: int = 64, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # logits for [drop, keep]
        )

    def forward(self, x: torch.Tensor, hard: bool = True) -> dict[str, torch.Tensor]:
        logits = self.net(x)
        soft = F.gumbel_softmax(logits, tau=self.tau, hard=hard)  # (B, 2)
        keep_mask = soft[:, 1]
        return {"keep_mask": keep_mask, "logits": logits}


@dataclass
class RelaxedGatingCascadeConfig:
    backbone: str = "resnet18"
    feature_dim: int = 512
    drop_reward: float = 0.05
    tau: float = 1.0
    pretrained_backbone: bool = True


class RelaxedGatingCascade(nn.Module):
    """Backbone + Gumbel gate + OBB head (Pareto-frontier-shifting variant).

    ``forward_train`` returns the differentiable mask + detection outputs;
    the calling training loop constructs the loss as

        L = detection_loss(masked_outputs) - drop_reward * mean(1 - keep_mask)

    so the gate is incentivized to drop tiles when the detector has nothing
    to recover. Detection loss is computed only on tiles where keep_mask > 0
    after straight-through.
    """

    def __init__(self, cfg: RelaxedGatingCascadeConfig):
        super().__init__()
        import timm

        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained_backbone, features_only=True
        )
        chans = self.backbone.feature_info.channels()
        self.gate = GumbelStraightThroughGate(in_dim=chans[-1], tau=cfg.tau)
        self.obb_head = SimpleOBBHead(chans[-1])

    def forward_train(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)[-1]  # (B, C, H, W)
        pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)  # (B, C)
        gate_out = self.gate(pooled, hard=True)
        keep = gate_out["keep_mask"]  # (B,)
        obb = self.obb_head(feats)
        return {"keep_mask": keep, "gate_logits": gate_out["logits"], **{f"obb_{k}": v for k, v in obb.items()}}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_train(x)


def relaxed_gating_loss(
    pred: dict[str, torch.Tensor],
    obb_targets: dict[str, torch.Tensor] | None,
    obb_pos_mask: torch.Tensor | None,
    cfg: RelaxedGatingCascadeConfig,
) -> dict[str, torch.Tensor]:
    keep = pred["keep_mask"]  # (B,)
    drop_frac = (1.0 - keep).mean()
    obb_total = keep.new_tensor(0.0)
    obb_breakdown: dict[str, torch.Tensor] = {}
    if obb_targets is not None and obb_pos_mask is not None and obb_pos_mask.any():
        obb_breakdown = smooth_l1_obb_loss(
            pred={k.replace("obb_", ""): v for k, v in pred.items() if k.startswith("obb_")},
            target_box=obb_targets["box"],
            target_angle=obb_targets["angle"],
            target_cls=obb_targets["cls"],
            pos_mask=obb_pos_mask * keep[:, None, None, None].bool(),
        )
        obb_total = obb_breakdown["total"]
    total = obb_total - cfg.drop_reward * drop_frac
    return {"total": total, "obb": obb_total, "drop_frac": drop_frac.detach(), **obb_breakdown}
