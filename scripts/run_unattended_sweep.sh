#!/usr/bin/env bash
# Unattended Phase 2.4: the 16-class sparsity sweep.
#
# One detector pass over the val tiles serves all 15 gate tasks, because the
# cascade composition is applied to cached predictions. Then one eval_cascade
# per gate task, restricted to that task's class via --gt-classes so mAP
# becomes that class's AP.
#
#   setsid nohup scripts/run_unattended_sweep.sh > logs/unattended_sweep.log 2>&1 </dev/null &

set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R"
PY="$R/.venv/bin/python"
DEV="${DEV:-1}"          # GPU 1 by default; GPU 0 is often the user's own work
LOGDIR="$R/logs/sweep"; mkdir -p "$LOGDIR"
echo $$ > "$R/logs/unattended_sweep.pid"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

WEIGHTS="$R/runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/weights/best.pt"
IMGROOT="$R/data/processed/dota_multiclass/images/val"
GT="$R/data/processed/dota_multiclass/labels/val"
PRED="$R/runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/predict_val"

log "sweep runner started (pid $$), device cuda:$DEV"
[ -f "$WEIGHTS" ] || { log "ABORT: missing $WEIGHTS"; exit 1; }

# --- 1. measured detector cost -------------------------------------------
GFLOPS_JSON="$R/reports/speed/yolo11m_multiclass_obb.json"
if [ ! -f "$GFLOPS_JSON" ]; then
  log "measuring detector GFLOPs/latency"
  $PY scripts/benchmark_speed.py --mode detector --weights "$WEIGHTS" \
      --imgsz 1024 --batch-size 8 --device "cuda:$DEV" --warmup 10 --iters 50 \
      --label yolo11m_multiclass_obb --out "$GFLOPS_JSON" \
      > "$LOGDIR/benchmark.log" 2>&1 || log "  WARNING: benchmark failed, will fall back"
fi
DET_G=$($PY -c "
import json,sys
try:
    print(json.load(open('$GFLOPS_JSON'))['gflops_per_image'])
except Exception:
    print(184.024825856)   # measured for the single-class YOLO11m-OBB at 1024
" 2>/dev/null)
log "detector GFLOPs/img = $DET_G"
GATE_G=0.5608915200   # mobilenetv3_large_100 @ 256, measured in reports/speed/

# --- 2. one detector pass over the val tiles ------------------------------
if [ ! -d "$PRED/labels" ]; then
  log "running detector over val tiles (save=False: the original pipeline's save=True wrote 40 GB of preview JPGs)"
  "$R/.venv/bin/yolo" predict task=obb model="$WEIGHTS" source="$IMGROOT" \
      save_txt=True save_conf=True save=False device="$DEV" imgsz=1024 \
      project="$R/runs/obb/runs/baseline_yolo11m_dota_multiclass_obb" \
      name=predict_val exist_ok=True > "$LOGDIR/predict_val.log" 2>&1
  if [ ! -d "$PRED/labels" ]; then
    log "ABORT: prediction labels not produced; see $LOGDIR/predict_val.log"; exit 1
  fi
fi
log "cached predictions: $(ls "$PRED/labels" | wc -l) tiles with detections"

# --- 3. per-class cascade sweeps ------------------------------------------
# class ids follow configs/datasets/dota_multiclass.yaml
declare -A CID=( [plane]=0 [ship]=1 [storage_tank]=2 [baseball_diamond]=3 \
  [tennis_court]=4 [basketball_court]=5 [ground_track_field]=6 [harbor]=7 \
  [bridge]=8 [large_vehicle]=9 [small_vehicle]=10 [helicopter]=11 \
  [roundabout]=12 [soccer_ball_field]=13 [swimming_pool]=14 [container_crane]=15 )
ALL_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

mkdir -p "$R/reports/sweep/cascade"
ok=0; fail=0
for f in "$R"/reports/gate_sweep_scores/*_val.jsonl; do
  task=$(basename "$f" _val.jsonl); task=${task#gate_sweep_}
  out="$R/reports/sweep/cascade/${task}.json"
  # --all-calibrations writes one file per calibration as <stem>_<calib>.json and
  # never --out itself, so test for those instead.
  if ls "$R/reports/sweep/cascade/${task}_"*.json >/dev/null 2>&1; then
    log "skip $task (done)"; continue
  fi
  if [ "$task" = "any" ]; then CLASSES="$ALL_IDS"; else CLASSES="${CID[$task]:-}"; fi
  if [ -z "$CLASSES" ]; then log "  SKIP $task (no class id)"; continue; fi
  log "eval_cascade $task (gt-classes: $CLASSES)"
  $PY scripts/eval_cascade.py \
      --scores "$f" --detector-runs "$PRED" --gt-labels "$GT" --image-root "$IMGROOT" \
      --gt-classes $CLASSES --all-calibrations \
      --metadata-jsonl "$R/data/processed/dota_multiclass/metadata/tiles.jsonl" \
      --gate-flops-g "$GATE_G" --detector-flops-g "$DET_G" \
      --label "gate_sweep_$task" --out "$out" > "$LOGDIR/eval_$task.log" 2>&1
  if ls "$R/reports/sweep/cascade/${task}_"*.json >/dev/null 2>&1; then
    n=$(ls "$R/reports/sweep/cascade/${task}_"*.json | wc -l)
    ok=$((ok+1)); log "  done $task ($n calibrations)"
  else fail=$((fail+1)); log "  FAILED $task (see $LOGDIR/eval_$task.log)"; fi
done
log "sweep finished: $ok ok, $fail failed"
exit 0
