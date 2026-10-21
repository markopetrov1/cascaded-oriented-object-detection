#!/bin/bash
# Evaluate the tap-depth study (WP9): each tap's head under the shared-backbone cost with its
# measured prefix and head GFLOPs, on the frozen ships detector's predictions.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SEED=${1:-67}
C=reports/r1/tap_scores/costs_seed$SEED.json
[ -f $C ] || { echo "no $C yet"; exit 0; }
for k in 2 4 6 8 10; do
  read P H < <($PY -c "import json; c=json.load(open('$C'))['taps']['$k']; print(c['prefix_gflops'], c['head_gflops'])")
  sfx=$([ "$SEED" = 67 ] && echo "" || echo "_seed$SEED")
  $PY scripts/r1_eval_cascade.py --preset tap --gate "tap$k=reports/r1/tap_scores/tap${k}_seed${SEED}_val.jsonl:0" \
      --fused-prefix $P --fused-head $H --out-name tap$k$sfx --workers 30
done
