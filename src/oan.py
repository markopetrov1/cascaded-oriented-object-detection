"""Faithful re-implementation of OAN's objectness head, on YOLO11-OBB.

Xie et al., "Fewer is More: Efficient Object Detection in Large Aerial Images"
(Sci. China Inf. Sci. 2023, arXiv:2212.13136) attach a small fully-convolutional
head to the detector's last backbone stage and train it jointly with detection.
Their reported design, which this file follows:

  * tap the last backbone stage (C5, stride 32);
  * 3x3 stride-2 convolution to 256 channels, then 1x1 to 512, then 1x1 to 1,
    giving a grid of objectness logits. At imgsz 1024 that is 16x16, i.e. one
    cell per 64x64 pixels, which is the grid size their Table 7 finds best;
  * a grid cell is positive when an object centre falls inside it;
  * focal loss on the grid, added to the detection loss as L = L_det + lambda *
    L_OAN with lambda in [3, 8];
  * a patch is forwarded when the maximum grid score clears a threshold chosen
    from training statistics, T = (m + v)^2 / k with k = 4, where m is the mean
    of per-map maxima and v the mean of per-map standard deviations over the
    final training iterations.

Why this exists: the paper's existing co-design experiments use an in-house
SimpleOBBHead on a ResNet-18 trunk, which is a weaker setup than OAN's and
therefore not a fair test of their claim that joint training is free. This
implements their architecture on the detector this paper actually uses, so the
comparison against an independently trained gate is like-for-like.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import OBBModel

# Index of the last backbone stage in the YOLO11 graph (C2PSA, stride 32).
# Layer 10 in ultralytics/cfg/models/11/yolo11-obb.yaml.
BACKBONE_TAP = 10


class OANHead(nn.Module):
    """Grid objectness head: 3x3 s2 -> 1x1 -> 1x1, as published."""

    def __init__(self, in_channels: int, mid: int = 256, hidden: int = 512) -> None:
        super().__init__()
        self.reduce = Conv(in_channels, mid, k=3, s=2)
        self.expand = Conv(mid, hidden, k=1, s=1)
        self.score = nn.Conv2d(hidden, 1, kernel_size=1)
        # Start strongly negative: the grid is overwhelmingly empty, and without
        # this the focal loss spends its first epochs undoing a 0.5 prior.
        nn.init.constant_(self.score.bias, -4.0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.score(self.expand(self.reduce(feat)))


def oan_targets(batch: dict[str, torch.Tensor], grid_h: int, grid_w: int,
                device: torch.device) -> torch.Tensor:
    """(B, 1, H, W) grid, 1 where an object centre falls in the cell.

    Uses the augmented batch's own normalised centres, so mosaic and flips stay
    consistent with the image the head sees.
    """
    batch_size = int(batch["img"].shape[0])
    target = torch.zeros((batch_size, 1, grid_h, grid_w), device=device)
    boxes = batch["bboxes"]
    if boxes.numel() == 0:
        return target
    idx = batch["batch_idx"].view(-1).long().to(device)
    cx = boxes[:, 0].to(device).clamp(0, 1 - 1e-6)
    cy = boxes[:, 1].to(device).clamp(0, 1 - 1e-6)
    gx = (cx * grid_w).long().clamp(0, grid_w - 1)
    gy = (cy * grid_h).long().clamp(0, grid_h - 1)
    target[idx, 0, gy, gx] = 1.0
    return target


def oan_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss averaged over grid cells (their 1/S^2 normalisation)."""
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    a_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (a_t * ce * (1.0 - p_t).pow(gamma)).mean()


class OANOBBModel(OBBModel):
    """YOLO11-OBB with OAN's objectness head trained jointly.

    The head is built eagerly in __init__ rather than on first forward, because
    the trainer constructs the optimizer from the parameters that exist at setup
    time; a lazily created head would silently never receive gradients.
    """

    def __init__(self, cfg: Any = "yolo11-obb.yaml", ch: int = 3, nc: int | None = None,
                 verbose: bool = True, oan_lambda: float = 3.0,
                 oan_tap: int = BACKBONE_TAP) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.oan_lambda = float(oan_lambda)
        self.oan_tap = int(oan_tap)
        self._oan_feat: torch.Tensor | None = None
        self.model[self.oan_tap].register_forward_hook(self._capture)

        # One dry forward to learn the tap's channel count, then build the head.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.forward(torch.zeros(1, ch, 256, 256))
        if self._oan_feat is None:
            raise RuntimeError(f"no feature captured at layer {self.oan_tap}")
        self.oan_head = OANHead(int(self._oan_feat.shape[1]))
        self._oan_feat = None
        if was_training:
            self.train()

    def _capture(self, _module, _inputs, output) -> None:
        self._oan_feat = output

    def loss(self, batch: dict[str, torch.Tensor], preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        total, items = self.criterion(preds, batch)

        feat = self._oan_feat
        if feat is None:  # should not happen once the hook is registered
            return total, items
        logits = self.oan_head(feat)
        targets = oan_targets(batch, logits.shape[-2], logits.shape[-1], logits.device)
        l_oan = oan_focal_loss(logits, targets)

        # The criterion returns one entry per loss component and the engine sums
        # them (trainer.py: self.loss = loss.sum()). So the OAN term is appended
        # as an extra component -- adding it as a scalar would broadcast across
        # every component and count it four times over. Scaled by batch size to
        # match the detection terms, which keeps lambda comparable to the paper's.
        batch_size = int(batch["img"].shape[0])
        oan_term = (self.oan_lambda * l_oan * batch_size).reshape(1)
        total = torch.cat([total.reshape(-1), oan_term])
        items = dict(items)
        items["oan"] = l_oan.detach()
        return total, items

    @torch.no_grad()
    def tile_scores(self, images: torch.Tensor) -> torch.Tensor:
        """Per-tile objectness = max over the grid, as OAN gates on."""
        self.forward(images)
        logits = self.oan_head(self._oan_feat)
        return logits.sigmoid().amax(dim=(1, 2, 3))


def oan_statistic_threshold(maps: list[torch.Tensor], k: float = 4.0) -> float:
    """Their threshold rule T = (m + v)^2 / k.

    m is the mean of per-map maxima and v the mean of per-map standard
    deviations, taken over the final training iterations. Reproduced here so it
    can be compared against temperature, Platt, isotonic and context-adaptive on
    the same footing rather than described in prose.
    """
    if not maps:
        raise ValueError("no activation maps supplied")
    maxima = torch.stack([m.sigmoid().amax() for m in maps])
    stds = torch.stack([m.sigmoid().std() for m in maps])
    m = float(maxima.mean())
    v = float(stds.mean())
    return (m + v) ** 2 / k
