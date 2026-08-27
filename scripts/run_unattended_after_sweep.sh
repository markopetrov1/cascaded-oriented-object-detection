#!/usr/bin/env bash
# Everything that can run without a human, once the 16-class sweep lands.
#
# Stage A (CPU): rebuild the law artifacts and every figure that depends on
#                them, now including the sweep's per-class points.
# Stage B (GPU): measured FLOPs and latency across batch sizes, which is what
#                the FLOPs-vs-wall-clock section needs instead of the current
#                (imgsz/1024)^2 proxy in wallclock_table.py.
#
#   setsid nohup scripts/run_unattended_after_sweep.sh > logs/unattended_after.log 2>&1 </dev/null &

set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R"
PY="$R/.venv/bin/python"
LOGDIR="$R/logs/after"; mkdir -p "$LOGDIR"
echo $$ > "$R/logs/unattended_after.pid"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run() {  # run <label> <cmd...>
  local label="$1"; shift
  log "-> $label"
  if "$@" > "$LOGDIR/$label.log" 2>&1; then
    log "   ok"
  else
    log "   FAILED (see $LOGDIR/$label.log)"
  fi
}

log "after-sweep runner started (pid $$)"

# --- wait for the sweep ----------------------------------------------------
if [ -f "$R/logs/unattended_sweep.pid" ]; then
  SWEEP_PID=$(cat "$R/logs/unattended_sweep.pid")
  waited=0
  while kill -0 "$SWEEP_PID" 2>/dev/null; do
    if [ $(( waited % 1800 )) -eq 0 ]; then
      log "waiting for sweep (pid $SWEEP_PID); $(ls "$R/reports/sweep/cascade" 2>/dev/null | wc -l)/15 tasks done"
    fi
    sleep 120; waited=$(( waited + 120 ))
    [ "$waited" -gt 86400 ] && { log "ABORT: sweep exceeded 24h"; exit 1; }
  done
fi
NDONE=$(ls "$R/reports/sweep/cascade" 2>/dev/null | wc -l)
log "sweep finished with $NDONE/15 tasks"
if [ "$NDONE" -eq 0 ]; then
  log "ABORT: sweep produced nothing; not rebuilding artifacts from an empty sweep"
  exit 1
fi

# --- Stage A: the law, with the sweep folded in ---------------------------
run savings_model        "$PY" scripts/savings_model.py
run sparsity_ceiling     "$PY" scripts/sparsity_ceiling.py
run object_loss_rate     "$PY" scripts/object_loss_rate.py
run prior_work           "$PY" scripts/prior_work_prediction.py
# figstyle lives in scripts/, so the figure scripts import it from there.
# Exported once rather than prefixed onto the `run` function, since assignment
# prefixes on shell functions are not portable.
export PYTHONPATH="$R/scripts"
run figures_law     "$PY" scripts/paper_figures_law.py
run figure_taxonomy "$PY" scripts/paper_figure_taxonomy.py
run figure_teaser   "$PY" scripts/paper_figure_teaser.py
unset PYTHONPATH

# --- Stage B: real FLOPs/latency across batch sizes ----------------------
pick_gpu() {
  for i in 1 0; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null | tr -d ' ')
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $i 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ $(( total - used )) -gt 10000 ] && { echo $i; return; }
  done
  echo ""
}
DEV=$(pick_gpu)
if [ -z "$DEV" ]; then
  log "no GPU with >10 GB free; skipping the latency sweep (re-run scripts/run_unattended_after_sweep.sh later)"
else
  log "latency sweep on cuda:$DEV"
  DETW="$R/runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/weights/best.pt"
  for b in 1 4 8 16; do
    run "speed_det_b$b" "$PY" scripts/benchmark_speed.py --mode detector \
      --weights "$DETW" --imgsz 1024 --batch-size $b --device "cuda:$DEV" \
      --warmup 10 --iters 50 --label "yolo11m_multiclass_obb_b$b" \
      --out "$R/reports/speed/yolo11m_multiclass_obb_b$b.json"
  done
  GW="$R/runs/gate_sweep_ship/best.pt"
  if [ -f "$GW" ]; then
    for b in 1 32 128; do
      run "speed_gate_b$b" "$PY" scripts/benchmark_speed.py --mode gate \
        --gate-config "$R/configs/experiments/_generated/gates/gate_sweep_ship.yaml" \
        --weights "$GW" --imgsz 256 --batch-size $b --device "cuda:$DEV" \
        --warmup 20 --iters 100 --label "gate_mbv3large_b$b" \
        --out "$R/reports/speed/gate_mbv3large_b$b.json"
    done
  fi
fi

run wallclock_table "$PY" scripts/wallclock_table.py

log "=== summary ==="
"$PY" scripts/savings_model.py 2>&1 | tail -25
log "after-sweep runner finished"
