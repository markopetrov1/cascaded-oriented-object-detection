#!/bin/bash
# Detector-size arm (Section IX, F13): YOLO11n detectors at input sizes 1024, 896 and 768 for
# ships, planes and small vehicles, each with (a) the headline rule over all six gate backbones
# and (b) the gate fixed to MobileNetV3-large, whose cost stays small next to YOLO11n's.
# Detector GFLOPs scale with the square of the input size.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
G=$($PY -c "import json; print(json.load(open('reports/r1/speed/components.json'))['detectors']['yolo11n_ships']['gflops'])")
declare -A GATE=([ship]=reports/gate_scores/gate_mbv3large_val.jsonl
                 [plane]=reports/planes/gate_scores/gate_planes_mbv3large_val.jsonl
                 ["small vehicle"]=reports/small_vehicle/gate_scores/gate_small_vehicle_mbv3large_val.jsonl)
declare -A SLUG=([ship]=ships [plane]=planes ["small vehicle"]=small_vehicle)
for task in ship plane "small vehicle"; do
  s_task=${task// /_}
  for s in 1024 896 768; do
    cache=runs/r1_cache/yolo11n_${SLUG[$task]}_${s}_val.npz
    g=$(python3 -c "print(round($G*($s/1024)**2, 4))")
    $PY scripts/r1_eval_cascade.py --preset yolo11n --tasks "$task" --cache $cache --det-gflops $g \
        --out-name ${s_task}_imgsz$s --workers 30
    $PY scripts/r1_eval_cascade.py --preset yolo11n --tasks "$task" --cache $cache --det-gflops $g \
        --gate "mbv3large=${GATE[$task]}" --out-name ${s_task}_imgsz${s}_mbv3large --workers 30
  done
done
