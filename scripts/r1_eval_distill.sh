#!/bin/bash
# F2: each redone distilled gate on its own under the held-out protocol (ships, 16-class detector).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for bb in mbv3small mbv3large resnet18 resnet50 effb0 tiny; do
  f=reports/r1/distill_scores/gate_ships_${bb}_distill_r1_val.jsonl
  [ -f $f ] && $PY scripts/r1_eval_cascade.py --preset distill --gate "$bb=$f" --out-name $bb --workers 30
done
f=reports/r1/distill_scores/gate_ships_mbv3large_distill_hard_r1_val.jsonl
[ -f $f ] && $PY scripts/r1_eval_cascade.py --preset distill --gate "mbv3large=$f" --out-name mbv3large_hard --workers 30
