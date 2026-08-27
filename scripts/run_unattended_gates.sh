#!/usr/bin/env bash
# Unattended Phase 2.3: per-class gates for the sparsity sweep.
#
# Waits for the multi-class detector to finish and for the gate-label trees to
# be built, then trains one MobileNetV3-large gate per gate task and scores the
# val tiles with it. Launch detached:
#
#   setsid nohup scripts/run_unattended_gates.sh > logs/unattended_gates.log 2>&1 </dev/null &
#
# Deliberately a separate script from run_unattended.sh: bash reads a script
# lazily by byte offset, so editing a running one corrupts its execution.

set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R"

LOGDIR="$R/logs/gate_sweep"
mkdir -p "$LOGDIR"
echo $$ > "$R/logs/unattended_gates.pid"

POLL_SECONDS=120
MAX_WAIT_HOURS=48

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "gate-sweep runner started (pid $$)"

# --- wait for the detector -------------------------------------------------
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
last_report=0
while :; do
  if ! pgrep -f "train_detector.py" > /dev/null 2>&1; then
    log "detector process no longer running"
    break
  fi
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    log "ABORT: detector still running after ${MAX_WAIT_HOURS}h"; exit 1
  fi
  if [ $(( now - last_report )) -ge 3600 ]; then
    log "still waiting for detector"
    last_report=$now
  fi
  sleep "$POLL_SECONDS"
done

# --- wait for the gate-label trees ----------------------------------------
while pgrep -f "make_gate_labels.py" > /dev/null 2>&1; do
  log "waiting for gate-label build"
  sleep "$POLL_SECONDS"
done

if [ ! -f "$R/reports/gate_tasks.json" ]; then
  log "ABORT: reports/gate_tasks.json missing"; exit 1
fi

log "generating gate configs"
"$R/.venv/bin/python" scripts/make_gate_configs.py 2>&1 | sed 's/^/    /'

CONFIGS=$(ls "$R/configs/experiments/_generated/gates/"*.yaml 2>/dev/null)
if [ -z "$CONFIGS" ]; then
  log "ABORT: no gate configs generated"; exit 1
fi
log "$(echo "$CONFIGS" | wc -l) gate tasks to train"

ok=0; fail=0
for cfg in $CONFIGS; do
  name=$(basename "$cfg" .yaml)
  if [ -f "$R/runs/$name/best.pt" ]; then
    log "skip $name (already trained)"
    continue
  fi
  log "training $name"
  "$R/.venv/bin/python" scripts/train_gate.py --config "$cfg" --device 0 \
    > "$LOGDIR/$name.log" 2>&1
  if [ $? -ne 0 ] || [ ! -f "$R/runs/$name/best.pt" ]; then
    log "  FAILED $name (see $LOGDIR/$name.log)"
    fail=$((fail+1))
    continue
  fi

  # Score val tiles so the cascade sweep has something to threshold.
  data_root=$(grep '^data_root:' "$cfg" | awk '{print $2}')
  "$R/.venv/bin/python" scripts/score_tiles.py \
    --weights "$R/runs/$name/best.pt" \
    --data-root "$data_root" \
    --split val \
    --out "$R/reports/gate_sweep_scores/${name}_val.jsonl" \
    --device 0 > "$LOGDIR/${name}_score.log" 2>&1
  if [ $? -eq 0 ]; then
    log "  done $name"
    ok=$((ok+1))
  else
    log "  trained but scoring FAILED $name"
    fail=$((fail+1))
  fi
done

log "gate sweep finished: $ok ok, $fail failed"
exit 0
