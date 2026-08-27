#!/usr/bin/env python3
"""Train YOLO11-OBB jointly with OAN's objectness head.

The counterpart to scripts/train_detector.py, for the co-design comparison. The
resulting model is a detector and a gate at once, which is exactly OAN's claim;
scripts/score_tiles_oan.py then emits per-tile scores in the same JSONL format
as the independently trained gates so eval_cascade can sweep both on identical
footing.

    python3 scripts/train_oan.py --data configs/datasets/dota_ships.yaml \
        --weights yolo11m-obb.pt --epochs 50 --lambda-oan 3.0 --device 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The user-global Ultralytics settings point runs_dir at another user's home
# (<REPO_ROOT> which is unreadable here. This redirects
# YOLO_CONFIG_DIR to a project-local file and must run before ultralytics is
# imported, since the settings are read at import time.
from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402

configure_ultralytics_settings()

from ultralytics.models.yolo.obb import OBBTrainer  # noqa: E402
from ultralytics.utils import RANK  # noqa: E402

from src.oan import OANOBBModel  # noqa: E402


class OANTrainer(OBBTrainer):
    """OBBTrainer that swaps in the OAN-augmented model.

    loss_names are derived by the engine from the criterion's loss dict, so the
    extra "oan" key appears in the progress table without further wiring.
    """

    oan_lambda = 3.0

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            OANOBBModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                oan_lambda=self.oan_lambda,
            )
        )
        if weights:
            model.load(weights)
        return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default="yolo11m-obb.pt")
    p.add_argument("--model-cfg", default="yolo11m-obb.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lambda-oan", type=float, default=3.0,
                   help="OAN loss weight; the paper uses 3 to 8 by detector")
    p.add_argument("--device", default="0")
    p.add_argument("--name", default=None)
    p.add_argument("--project", default="runs/oan")
    p.add_argument("--seed", type=int, default=67)
    args = p.parse_args()

    name = args.name or f"oan_lambda{args.lambda_oan:g}"
    OANTrainer.oan_lambda = args.lambda_oan
    trainer = OANTrainer(overrides={
        "model": args.model_cfg,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "device": args.device,
        "seed": args.seed,
        "project": args.project,
        "name": name,
        "exist_ok": True,
        "pretrained": args.weights,
        "amp": True,
        "val": True,
        # In training, `save` gates model checkpointing, not image saving --
        # `save: False` here produced 25 epochs with an empty weights/ directory.
        # The tens of GB of preview JPGs in runs/ come from `predict save=True`,
        # which is a different flag on a different mode.
        "save": True,
        "plots": False,
    })
    trainer.train()
    # Ultralytics resolves `project` against its own runs_dir setting, so the
    # save directory is not simply <project>/<name>. Record the resolved path so
    # downstream scripts read it instead of guessing.
    marker = Path(__file__).resolve().parents[1] / "runs" / "oan_last_save_dir.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(trainer.save_dir))
    print(f"[train_oan] done -> {trainer.save_dir}")
    print(f"[train_oan] path recorded in {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
