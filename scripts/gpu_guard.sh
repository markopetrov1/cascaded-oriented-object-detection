#!/bin/bash
# Refuse to start project GPU work while the target device is busy with a job
# this project did not start.
#
# "Someone else" means any compute process on the device that is not this
# project's own ollama server: another user, or one of this account's own
# training runs, which is the case that actually bites. On 2026-08-28 GPU 0 was
# held by this account's own PyTorch jobs at 14.3 GB and full utilisation while
# GPU 1 sat idle, which is exactly the collision this guard exists to prevent.
#
#   ./scripts/gpu_guard.sh                  # checks $PROJECT_GPU, default 0
#   PROJECT_GPU=1 ./scripts/gpu_guard.sh
#
# Exit 0 when it is safe to run, 1 when the device is taken or cannot be read.
#
# Every nvidia-smi call is wrapped in a timeout. On 2026-08-28 the process query
# began emitting the first device's rows and then blocking forever, which hung
# this guard and would have hung every caller with it. A guard that cannot read
# the device refuses rather than assuming the device is free.
PROJECT_GPU="${PROJECT_GPU:-0}"
SMI_TIMEOUT="${SMI_TIMEOUT:-10}"

UUID=$(timeout "$SMI_TIMEOUT" nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
       | awk -F', ' -v g="$PROJECT_GPU" '$1==g{print $2}')
if [ -z "$UUID" ]; then
  echo "cannot read GPU $PROJECT_GPU (nvidia-smi timed out or no such device)"
  exit 1
fi

APPS=$(timeout "$SMI_TIMEOUT" nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid \
       --format=csv,noheader 2>/dev/null)
SMI_RC=$?
if [ $SMI_RC -ne 0 ]; then
  # The query hung or failed. Fall back to the memory reading, which is a
  # separate and so far reliable code path, and treat a loaded device as taken.
  USED=$(timeout "$SMI_TIMEOUT" nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v g="$PROJECT_GPU" '$1==g{print $2}')
  echo "warning: process query failed (rc=$SMI_RC); falling back to memory reading"
  if [ -z "$USED" ]; then echo "cannot read GPU $PROJECT_GPU memory either."; exit 1; fi
  # A few GB is the idling system ollama and the root vLLM engine, both of which
  # yield on demand. Anything much above that is a real tenant.
  if [ "$USED" -gt "${FREE_MB:-6000}" ]; then
    echo "GPU $PROJECT_GPU holds ${USED} MiB, above the ${FREE_MB:-6000} MiB idle allowance."
    echo "REFUSING to start: GPU $PROJECT_GPU looks taken."
    exit 1
  fi
  echo "GPU $PROJECT_GPU is free (${USED} MiB resident, within the idle allowance)."
  exit 0
fi

# Match the executable name, never the full command line. `pgrep -f` matched a
# PyTorch training job of this account's on 2026-08-28 because "llama" appeared
# somewhere in its arguments, so the guard skipped an 18.6 GB tenant and called
# the device free. Only a process actually named ollama or llama-server is ours.
MINE=$(timeout 10 ps -o pid=,comm= -u "$(whoami)" 2>/dev/null \
       | awk '$2=="ollama" || $2=="llama-server" {print $1}' \
       | tr '\n' '|' | sed 's/|$//')
# A stage that loads a model directly rather than through ollama is still this
# project's own work. Such callers export MINE_PGID with their process group so
# the guard does not mistake their own child for somebody else's job.
MINE_PGID="${MINE_PGID:-}"
BUSY=0
while IFS=, read -r pid mem uuid; do
  pid=$(echo "$pid" | tr -d ' '); uuid=$(echo "$uuid" | tr -d ' ')
  [ -n "$pid" ] || continue
  [ "$uuid" = "$UUID" ] || continue
  if [ -n "$MINE" ] && echo "$pid" | grep -qE "^($MINE)$"; then continue; fi
  if [ -n "$MINE_PGID" ]; then
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$pgid" = "$MINE_PGID" ] && continue
  fi
  read -r user cmd <<< "$(ps -o user=,comm= -p "$pid" 2>/dev/null)"
  # The system-wide ollama service and the root vLLM engine both idle at a few
  # GB and yield on demand, so neither counts as the device being taken.
  if [ "$user" = "ollama" ] || [ "$user" = "root" ]; then continue; fi
  echo "GPU $PROJECT_GPU busy: pid $pid (${user:-unknown}, ${cmd:-unknown}) using$mem"
  BUSY=1
done <<< "$APPS"

if [ "$BUSY" = "1" ]; then
  # Sharing a device is sometimes the right call: a tenant holding memory while
  # sitting at a few percent utilisation is not really using it, and waiting for
  # it costs more than the contention would. That is a judgement about someone
  # else's work, so it is never made automatically. ALLOW_SHARED=1 says the
  # operator looked at the device and decided, and the utilisation is printed so
  # the decision is on the record.
  if [ "${ALLOW_SHARED:-0}" = "1" ]; then
    UTIL=$(timeout "$SMI_TIMEOUT" nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
           --format=csv,noheader,nounits | awk -F', ' -v g="$PROJECT_GPU" '$1==g{print $2"% util, "$3" MiB"}')
    echo "ALLOW_SHARED set: proceeding on a device held by someone else ($UTIL)."
    exit 0
  fi
  echo "REFUSING to start: GPU $PROJECT_GPU is taken."
  echo "Set ALLOW_SHARED=1 to share it deliberately."
  exit 1
fi
echo "GPU $PROJECT_GPU is free."
exit 0
