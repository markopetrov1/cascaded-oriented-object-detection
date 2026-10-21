#!/bin/bash
# F3: the fixed joint-training gates and the independent ResNet-18 under the held-out protocol.
# GFLOPs per tile of each gate path come from reports/r1/codesign_gflops.json (r1_codesign_cost.py).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
G=reports/r1/codesign_gflops.json
[ -f $G ] || $PY scripts/r1_codesign_cost.py
for v in shared early_exit; do
  f=reports/r1/codesign_scores/r1_codesign_${v}_resnet18_val.jsonl
  gf=$($PY -c "import json; print(json.load(open('$G'))['$v'])")
  [ -f $f ] && $PY scripts/r1_eval_cascade.py --preset codesign --gate "$v=$f:$gf" --out-name $v --workers 30
done
$PY scripts/r1_eval_cascade.py --preset codesign --gate "resnet18=reports/gate_scores/gate_resnet18_val.jsonl" \
    --out-name resnet18 --workers 30
