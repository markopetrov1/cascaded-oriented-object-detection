#!/bin/bash
# SODA-A transfer (WP10, R2.6a): cache from the in-memory-tiled predictions, then the DOTA-chosen
# operating points of the sweep (MobileNetV3-large) and the three-class (six backbones) arms.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PRED=runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/predict_soda_val_c001
[ -f $PRED/r1_predict_meta.json ] || { echo "SODA-A predictions not ready"; exit 0; }
[ -f runs/r1_cache/multiclass_soda.npz ] || $PY scripts/r1_build_cache.py --soda --pred $PRED --out runs/r1_cache/multiclass_soda.npz
$PY scripts/r1_eval_cascade.py --preset soda_sweep --workers 30
$PY scripts/r1_eval_cascade.py --preset soda_pillars --workers 30
