#!/usr/bin/env bash
# Unattended Phase 2 runner.
#
# Waits until GPU 0 is free of everyone else's work, then trains the
# multi-class YOLO11m-OBB detector. Designed to be launched detached
# (setsid + nohup + </dev/null) so it keeps running after logout.
#
#   setsid nohup scripts/run_unattended.sh > logs/unattended.log 2>&1 </dev/null &
#
# Status:  tail -f logs/unattended.log
# Stop:    kill $(cat logs/unattended.pid)

set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R"

LOGDIR="$R/logs/multiclass"
mkdir -p "$LOGDIR"
echo $$ > "$R/logs/unattended.pid"

# GPU 0 counts as free below this many MiB; a truly idle card sits near 1 MiB,
# and this leaves room for driver overhead without tolerating a real job.
FREE_MIB=500
POLL_SECONDS=60
MAX_WAIT_HOURS=72

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "unattended runner started (pid $$)"
log "waiting for GPU 0 to fall below ${FREE_MIB} MiB, polling every ${POLL_SECONDS}s"

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
last_report=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  if [ -n "$used" ] && [ "$used" -lt "$FREE_MIB" ]; then
    log "GPU 0 clear (${used} MiB). Proceeding."
    break
  fi
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    log "ABORT: GPU 0 still busy (${used} MiB) after ${MAX_WAIT_HOURS}h"
    exit 1
  fi
  # Heartbeat every 30 min so the log shows the watcher is alive without
  # producing a line a minute for hours.
  if [ $(( now - last_report )) -ge 1800 ]; then
    log "still waiting; GPU 0 at ${used} MiB"
    last_report=$now
  fi
  sleep "$POLL_SECONDS"
done

# Guard against launching on labels that are incomplete or, worse, the
# single-class tree a directory symlink used to redirect us to.
NTRAIN=$(ls "$R/data/processed/dota_multiclass/labels/train" 2>/dev/null | wc -l)
NVAL=$(ls "$R/data/processed/dota_multiclass/labels/val" 2>/dev/null | wc -l)
log "label check: train=$NTRAIN val=$NVAL"
if [ "$NTRAIN" -lt 15749 ] || [ "$NVAL" -lt 5297 ]; then
  log "ABORT: incomplete multiclass labels"
  exit 1
fi

RESOLVED=$("$R/.venv/bin/python" - <<'PY'
import glob
from ultralytics.data.utils import img2label_paths
imgs = sorted(glob.glob("data/processed/dota_multiclass/images/val/*.png"))[:1]
print(img2label_paths(imgs)[0] if imgs else "NONE")
PY
)
log "label path resolves to: $RESOLVED"
case "$RESOLVED" in
  *dota_multiclass/labels/*) ;;
  *) log "ABORT: labels resolve outside dota_multiclass ($RESOLVED)"; exit 1 ;;
esac

log "launching multi-class YOLO11m-OBB (16 classes, patience=50) on GPU 0"
"$R/.venv/bin/python" scripts/train_detector.py \
  --config configs/experiments/baseline_yolo11m_multiclass_obb.yaml \
  --device 0 > "$LOGDIR/train_yolo11m.log" 2>&1
status=$?
log "detector training exited with status $status"

if [ "$status" -eq 0 ]; then
  BEST="$R/runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/weights/best.pt"
  [ -f "$BEST" ] && log "best weights: $BEST" || log "WARNING: expected weights missing at $BEST"
fi

log "unattended runner finished"
exit $status
