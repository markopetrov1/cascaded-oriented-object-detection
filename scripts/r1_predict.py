#!/usr/bin/env python3
"""Cache detector predictions without truncating the precision-recall curve.

`yolo predict` keeps only detections above conf 0.25 and at most 300 per image,
and every cascade evaluation in the first submission was computed from caches
made that way, so each AP was integrated over a truncated curve. This writes the
same label layout as `predict save_txt=True save_conf=True` (one file per tile,
rows `cls x1 y1 x2 y2 x3 y3 x4 y4 conf`, corners normalised by the tile size) but
at conf 0.001 and up to 1000 detections, which is the setting Ultralytics uses for
validation.

Resumable: tiles that already have a label file are skipped, and a tile with no
detections gets an empty file so it is not redone.

    python scripts/r1_predict.py --weights <best.pt> \
        --source data/processed/dota_multiclass/images/val \
        --out runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/predict_val_c001 --device 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must precede the ultralytics import (settings are read at import time). It
# also puts the repository on sys.path, which unpickling the OAN checkpoint needs.
from src.utils.ultralytics_env import configure_ultralytics_settings  # noqa: E402

configure_ultralytics_settings()

from ultralytics import YOLO  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="directory of tile images")
    p.add_argument("--out", required=True, help="run directory; labels/ is written inside")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU, the Ultralytics default")
    # 10000, not Ultralytics' validation default of 300 or the 1000 first tried
    # here: a few dense DOTA tiles (harbours, parking lots) carry more than 1000
    # detections above conf 0.25, so a cap of 1000 still truncated them.
    p.add_argument("--max-det", type=int, default=10000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="1")
    p.add_argument("--redo-capped", type=int, default=0,
                   help="re-predict only tiles whose existing label file has at least this many "
                        "rows (i.e. hit an earlier max_det), overwriting them")
    args = p.parse_args()

    out = Path(args.out)
    labels = out / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    sources = sorted(q for q in Path(args.source).iterdir()
                     if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    if args.redo_capped:
        def n_rows(f: Path) -> int:
            with open(f) as fh:
                return sum(1 for _ in fh)
        capped = {f.stem for f in labels.glob("*.txt") if n_rows(f) >= args.redo_capped}
        todo = [s for s in sources if s.stem in capped]
        done = set()
    else:
        done = {f.stem for f in labels.glob("*.txt")}
        todo = [s for s in sources if s.stem not in done]
    print(f"[r1_predict] {len(sources)} images, {len(done)} already done, {len(todo)} to go", flush=True)

    model = YOLO(args.weights, task="obb")
    t0 = time.time()
    n_det = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        results = model.predict([str(c) for c in chunk], imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, max_det=args.max_det, device=args.device,
                                verbose=False, save=False)
        for src, res in zip(chunk, results):
            rows = []
            obb = res.obb
            if obb is not None and len(obb):
                corners = obb.xyxyxyxyn.cpu().numpy().reshape(-1, 8)
                cls = obb.cls.cpu().numpy().astype(int)
                conf = obb.conf.cpu().numpy()
                rows = [f"{c} " + " ".join(f"{v:.6f}" for v in xy) + f" {s:.6f}"
                        for c, xy, s in zip(cls, corners, conf)]
                n_det += len(rows)
            (labels / f"{src.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        if (i // args.batch) % 100 == 0:
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

    meta_path = out / ("r1_predict_meta.json" if not args.redo_capped else "r1_predict_recap_meta.json")
    meta = {"weights": args.weights, "source": args.source, "imgsz": args.imgsz,
            "conf": args.conf, "iou": args.iou, "max_det": args.max_det,
            "n_images": len(sources), "seconds_this_run": round(time.time() - t0, 1),
            "detections_this_run": n_det, "redo_capped": args.redo_capped,
            "tiles_this_run": len(todo)}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[r1_predict] done: {meta}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
