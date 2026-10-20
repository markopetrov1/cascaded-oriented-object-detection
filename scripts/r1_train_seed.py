#!/usr/bin/env python3
"""Repeat the fused-versus-independent ships comparison at another seed.

Both arms copy the settings of the seed-67 runs from the first submission
(runs/obb/runs/oan/oan_ships_lambda3 and runs/obb/runs/baseline_yolo11m_dota_ships_obb_p50):
DOTA-pretrained yolo11m-obb weights, 50 epochs, patience 50, batch 16, imgsz
1024, AMP, deterministic, workers 8, SGD 'auto'. Only the seed and the output
directory change.

  --kind oan      YOLO11m-OBB trained jointly with the OAN objectness head (lambda 3)
  --kind control  plain YOLO11m-OBB, the matched detector for the independent gate

Resumable: if the run directory has weights/last.pt and no completion marker,
training resumes from it, which is what the queue watchdog relies on after a
preemption. A finished run (marker present) is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402

configure_ultralytics_settings()

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.obb import OBBTrainer  # noqa: E402
from ultralytics.utils import RANK  # noqa: E402

from src.oan import OANOBBModel  # noqa: E402

PROJECT = REPO / "runs" / "obb" / "runs" / "r1_seeds"
DATA = "configs/datasets/dota_ships.yaml"
COMMON = dict(data=DATA, epochs=50, patience=50, batch=16, imgsz=1024, workers=8,
              deterministic=True, amp=True, val=True, plots=False, save=True,
              exist_ok=True, optimizer="auto")


class OANTrainer(OBBTrainer):
    """Same as scripts/train_oan.py: an OBBTrainer that builds the OAN model."""

    oan_lambda = 3.0

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            OANOBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"],
                        verbose=verbose and RANK == -1, oan_lambda=self.oan_lambda))
        if weights:
            model.load(weights)
        return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("oan", "control"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="1")
    args = p.parse_args()

    name = (f"oan_ships_lambda3_seed{args.seed}" if args.kind == "oan"
            else f"baseline_yolo11m_dota_ships_obb_p50_seed{args.seed}")
    run = PROJECT / name
    marker = run / "r1_done.json"
    if marker.exists():
        print(f"[r1_train_seed] {name} already finished: {marker}")
        return 0
    last = run / "weights" / "last.pt"

    if args.kind == "oan":
        if last.exists():
            print(f"[r1_train_seed] resuming {name} from {last}", flush=True)
            trainer = OANTrainer(overrides={"resume": str(last), "device": args.device})
        else:
            trainer = OANTrainer(overrides=dict(COMMON, model="yolo11m-obb.yaml",
                                                pretrained="yolo11m-obb.pt", device=args.device,
                                                seed=args.seed, project=str(PROJECT), name=name))
        trainer.train()
        save_dir = Path(trainer.save_dir)
    else:
        if last.exists():
            print(f"[r1_train_seed] resuming {name} from {last}", flush=True)
            model = YOLO(str(last))
            model.train(resume=True, device=args.device)
        else:
            model = YOLO("yolo11m-obb.pt")
            model.train(**COMMON, device=args.device, seed=args.seed,
                        project=str(PROJECT), name=name, pretrained=True)
        save_dir = run

    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"[r1_train_seed] no best.pt in {save_dir}", flush=True)
        return 1
    marker.write_text(json.dumps({"kind": args.kind, "seed": args.seed,
                                  "save_dir": str(save_dir), "best": str(best)}, indent=2))
    print(f"[r1_train_seed] finished {name} -> {save_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
