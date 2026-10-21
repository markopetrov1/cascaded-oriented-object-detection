#!/usr/bin/env python3
"""GFLOPs per tile (256 px, thop, batch 1) of the gate path of each joint-training variant.

The shared-backbone gate reads the last stage of its ResNet-18 trunk, so its gate path is
the whole trunk plus a pooled linear head. The early-exit gate reads the trunk after its
first residual stage (exit_at = 1), so its gate path is the stem, that stage and the head.
Writes reports/r1/codesign_gflops.json.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from thop import profile

REPO = Path(__file__).resolve().parents[1]


def gflops(m: nn.Module) -> float:
    macs, _ = profile(copy.deepcopy(m).eval(), inputs=(torch.zeros(1, 3, 256, 256),), verbose=False)
    return 2 * macs / 1e9


def main() -> int:
    r = timm.create_model("resnet18", pretrained=False)
    head = lambda c: nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 1))
    shared = nn.Sequential(r.conv1, r.bn1, r.act1, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4, head(512))
    early = nn.Sequential(r.conv1, r.bn1, r.act1, r.maxpool, r.layer1, head(64))
    out = {"shared": gflops(shared), "early_exit": gflops(early), "note": "gate path only, 256 px, batch 1"}
    (REPO / "reports/r1/codesign_gflops.json").write_text(json.dumps(out, indent=2))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
