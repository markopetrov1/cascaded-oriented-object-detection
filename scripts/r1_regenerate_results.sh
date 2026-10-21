#!/bin/bash
# The round-1 evaluation chain, in order, from cached detector predictions and gate scores to
# reports/r1. GPU steps (predictions, training, timing) ran through the queue in runs/r1_queue
# (job scripts kept in runs/r1_queue/done); this script repeats the CPU steps. Every random
# choice (folds, bootstrap resamples) is seeded, so a rerun reproduces reports/r1.
#
#   scripts/r1_regenerate_results.sh          # all CPU steps (several hours on 30 cores)
#   scripts/r1_regenerate_paper.sh            # then the manuscript's numbers, tables, figures
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=runs/obb/runs

# 1. Image-level evaluation caches (scripts/r1_build_cache.py)
$PY scripts/r1_build_cache.py --pred $R/baseline_yolo11m_dota_multiclass_obb/predict_val_c001 --out runs/r1_cache/multiclass_val.npz
$PY scripts/r1_build_cache.py --pred $R/baseline_yolo11m_dota_ships_obb_p50/predict_val_c001 --class-map 0:1 --out runs/r1_cache/shipsp50_val.npz
$PY scripts/r1_build_cache.py --pred $R/oan/oan_ships_lambda3/predict_val_c001 --class-map 0:1 --out runs/r1_cache/oan_val.npz
$PY scripts/r1_build_cache.py --pred $R/baseline_yolo11m_dota_ships_obb_p50/predict_hrsc_test_c001 --hrsc --out runs/r1_cache/shipsp50_hrsc.npz
for s in 1024 896 768; do
  $PY scripts/r1_build_cache.py --pred $R/baseline_yolo11n_dota_ships_obb/predict_val_c001_imgsz$s --class-map 0:1 --out runs/r1_cache/yolo11n_ships_${s}_val.npz
  $PY scripts/r1_build_cache.py --pred $R/baseline_yolo11n_dota_planes_obb/predict_val_c001_imgsz$s --class-map 0:0 --out runs/r1_cache/yolo11n_planes_${s}_val.npz
  $PY scripts/r1_build_cache.py --pred $R/baseline_yolo11n_dota_small_vehicle_obb/predict_val_c001_imgsz$s --class-map 0:10 --out runs/r1_cache/yolo11n_small_vehicle_${s}_val.npz
done

# 2. Scene folds (seeded) and the held-out cascade evaluations
$PY scripts/r1_make_folds.py
for p in indep fused hrsc shipsp50 pillars sweep; do $PY scripts/r1_eval_cascade.py --preset $p --workers 30; done
scripts/r1_run_yolo11n.sh
scripts/r1_eval_soda.sh            # SODA-A (data prepared by r1_fetch_soda_val.py, r1_prepare_soda.py, r1_soda_infer.py)
scripts/r1_eval_codesign.sh
scripts/r1_eval_distill.sh
scripts/r1_eval_taps.sh 67; scripts/r1_eval_taps.sh 1; scripts/r1_eval_taps.sh 2
scripts/r1_eval_seeds.sh

# 3. Object-level recall, strata, calibration ties, prior work, overlap check
for r in pillars/ship pillars/plane pillars/small_vehicle indep/ship fused/ship shipsp50/ship; do
  $PY scripts/r1_object_recall.py --run reports/r1/cascade/$r.json --masks risk_pooled risk_full_pooled plain_full_pooled same_split
done
for r in pillars/ship pillars/plane pillars/small_vehicle; do $PY scripts/r1_stratified.py --run reports/r1/cascade/$r.json; done
$PY scripts/r1_strata_scenes.py
$PY scripts/r1_calibration_ties.py --task ship
$PY scripts/r1_dota10_empty_rate.py
$PY scripts/r1_soda_dota_overlap.py
$PY scripts/r1_codesign_cost.py
$PY scripts/r1_distill_targets.py

# 4. Timing: merge the clean benchmark runs (runs 3 and 4, after the set_device fix)
$PY scripts/r1_merge_benchmarks.py reports/r1/speed/components_run3.json reports/r1/speed/components_run4.json
