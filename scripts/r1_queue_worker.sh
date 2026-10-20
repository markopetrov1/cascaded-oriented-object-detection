#!/bin/bash
# File-based GPU job queue for the round-1 revision runs.
#
#   setsid nohup scripts/r1_queue_worker.sh 1 > runs/r1_queue/worker1.log 2>&1 &
#
# Jobs are executable scripts in runs/r1_queue/pending/, taken in name order
# (prefix them NN_). A worker owns one card. Before a job starts, and every
# GUARD_POLL seconds while it runs, scripts/gpu_guard.sh checks that no process
# of someone else's holds the card; if one appears, the job's process group is
# stopped with SIGINT (Ultralytics saves last.pt on it), the job goes back to
# pending, and the worker waits for the card. Jobs must therefore be resumable.
#
# A job whose script contains the line "# EXCLUSIVE" (timing measurements) only
# starts when no other queue job is running, and while it waits at the head of
# the queue no worker starts anything else.
#
# A job may pin a card with "# GPU: <n>". The job sees its card as
# $R1_GPU and CUDA_VISIBLE_DEVICES is left alone, so scripts pass device=$R1_GPU.
# Touch runs/r1_queue/STOP to make workers exit after their current job.
set -u
GPU="${1:?usage: r1_queue_worker.sh <gpu index>}"
R="$(cd "$(dirname "$0")/.." && pwd)"
Q="$R/runs/r1_queue"
GUARD_POLL="${GUARD_POLL:-60}"
IDLE_POLL="${IDLE_POLL:-300}"
mkdir -p "$Q"/{pending,running,done,failed,logs}
log() { echo "[$(date '+%F %T')] [gpu$GPU] $*"; }

card_free() { PROJECT_GPU="$GPU" MINE_PGID="${1:-}" "$R/scripts/gpu_guard.sh" >/dev/null 2>&1; }

next_job() {
  local f
  for f in $(ls "$Q/pending" 2>/dev/null | sort); do
    local path="$Q/pending/$f"
    local pin; pin=$(grep -m1 '^# GPU:' "$path" | awk '{print $3}')
    if grep -q '^# EXCLUSIVE' "$path"; then
      # Head-of-queue exclusive job: run it only when nothing else is running,
      # and block everything behind it meanwhile.
      [ -n "$(ls "$Q/running" 2>/dev/null)" ] && return 1
      { [ -z "$pin" ] || [ "$pin" = "$GPU" ]; } || return 1
      echo "$f"; return 0
    fi
    if [ -z "$pin" ] || [ "$pin" = "$GPU" ]; then echo "$f"; return 0; fi
  done
  return 1
}

log "worker started (pid $$)"
while true; do
  [ -f "$Q/STOP" ] && { log "STOP present, exiting"; exit 0; }
  job=$(next_job) || { sleep "$IDLE_POLL"; continue; }
  if ! card_free; then log "card busy, waiting before $job"; sleep "$IDLE_POLL"; continue; fi
  mv "$Q/pending/$job" "$Q/running/$job" 2>/dev/null || continue   # another worker took it
  log "start $job"
  echo "$(date '+%F %T') start gpu$GPU" >> "$Q/logs/$job.history"
  R1_GPU="$GPU" setsid bash "$Q/running/$job" >> "$Q/logs/$job.log" 2>&1 &
  jpid=$!
  sleep 2
  jpgid=$(ps -o pgid= -p "$jpid" 2>/dev/null | tr -d ' ')
  echo "$jpid $jpgid gpu$GPU $(date '+%F %T')" > "$Q/running/$job.pid"
  preempted=0
  while kill -0 "$jpid" 2>/dev/null; do
    sleep "$GUARD_POLL"
    kill -0 "$jpid" 2>/dev/null || break
    if ! card_free "$jpgid"; then
      log "foreign job on gpu$GPU, stopping $job (pgid $jpgid)"
      kill -INT -- "-$jpgid" 2>/dev/null
      for _ in $(seq 1 24); do kill -0 "$jpid" 2>/dev/null || break; sleep 5; done
      kill -TERM -- "-$jpgid" 2>/dev/null
      preempted=1
      break
    fi
  done
  wait "$jpid" 2>/dev/null; rc=$?
  rm -f "$Q/running/$job.pid"
  if [ "$preempted" = 1 ]; then
    mv "$Q/running/$job" "$Q/pending/$job"
    echo "$(date '+%F %T') preempted" >> "$Q/logs/$job.history"
    log "requeued $job; waiting for the card"
    sleep "$IDLE_POLL"
  elif [ "$rc" = 0 ]; then
    mv "$Q/running/$job" "$Q/done/$job"
    echo "$(date '+%F %T') done" >> "$Q/logs/$job.history"
    log "done $job"
  else
    mv "$Q/running/$job" "$Q/failed/$job"
    echo "$(date '+%F %T') failed rc=$rc" >> "$Q/logs/$job.history"
    log "FAILED $job (rc=$rc), see $Q/logs/$job.log"
  fi
done
