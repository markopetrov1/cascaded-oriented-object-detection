#!/bin/bash
# Seed repeats of the fused-versus-decoupled comparison (R1.6, R2.4). For seed N in {1, 2}:
# the jointly trained OAN detector's predictions and head scores give the fused arm; the
# control detector trained with the same seed plus the independent MobileNetV3-large gate
# trained with that seed give the decoupled arm. Seed 67 is the original pair
# (presets fused and indep). Runs whatever seeds have finished; safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in 1 2; do
  ofile=runs/obb/runs/r1_seeds/oan_ships_lambda3_seed$s/r1_done.json
  cfile=runs/obb/runs/r1_seeds/baseline_yolo11m_dota_ships_obb_p50_seed$s/r1_done.json
  oscore=reports/r1/oan_seeds/gate_oan_ships_seed${s}_val.jsonl
  gscore=reports/r1/gate_scores/gate_mbv3large_seed${s}_val.jsonl
  if [ -f $ofile ] && [ -f $oscore ]; then
    D=$($PY -c "import json;print(json.load(open('$ofile'))['save_dir'])")
    if [ -f $D/predict_val_c001/r1_predict_meta.json ]; then
      [ -f runs/r1_cache/oan_seed${s}_val.npz ] || $PY scripts/r1_build_cache.py --pred $D/predict_val_c001 \
          --out runs/r1_cache/oan_seed${s}_val.npz --class-map 0:1
      $PY scripts/r1_eval_cascade.py --preset fused --cache runs/r1_cache/oan_seed${s}_val.npz \
          --gate "oan_head=$oscore:0" --out-name ship_seed$s --workers 30
    fi
  else echo "fused seed $s not ready"; fi
  if [ -f $cfile ] && [ -f $gscore ]; then
    D=$($PY -c "import json;print(json.load(open('$cfile'))['save_dir'])")
    if [ -f $D/predict_val_c001/r1_predict_meta.json ]; then
      [ -f runs/r1_cache/shipsp50_seed${s}_val.npz ] || $PY scripts/r1_build_cache.py --pred $D/predict_val_c001 \
          --out runs/r1_cache/shipsp50_seed${s}_val.npz --class-map 0:1
      $PY scripts/r1_eval_cascade.py --preset indep --cache runs/r1_cache/shipsp50_seed${s}_val.npz \
          --gate "mbv3large=$gscore" --out-name ship_seed$s --workers 30
    fi
  else echo "decoupled seed $s not ready"; fi
done
