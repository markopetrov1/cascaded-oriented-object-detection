#!/usr/bin/env bash
# Run eval_cascade for planes/tiny gate (4 calibrations) after tiny_val scoring completes.
# Run this AFTER planes_resume pipeline finishes.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROCESSED="data/processed/dota_planes"
DET_RUN_DIR="runs/obb/runs/baseline_yolo11m_dota_planes_obb"
cascade_dir="reports/planes/cascade"

scores="reports/planes/gate_scores/gate_planes_tiny_val.jsonl"
if [ ! -f "$scores" ] || [ "$(wc -l < "$scores")" -lt 100 ]; then
    echo "ERROR: $scores not ready yet. Run score_tiles for tiny/val first."
    exit 1
fi

for calib in identity temperature platt isotonic; do
    out="$cascade_dir/gate_planes_tiny_${calib}.json"
    if [ -f "$out" ]; then
        echo "SKIP $out (exists)"
        continue
    fi
    echo "Running eval_cascade for planes/tiny/$calib..."
    python3 scripts/eval_cascade.py \
        --scores "$scores" \
        --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
        --detector-runs "${DET_RUN_DIR}/predict_val" \
        --gt-labels "$PROCESSED/labels/val" \
        --image-root "$PROCESSED/images/val" \
        --calibration "$calib" \
        --gate-flops-g 1.8 --gate-latency-ms 2.0 \
        --detector-flops-g 91 --detector-latency-ms 22 \
        --label "gate_planes_tiny" \
        --out "$out"
    echo "Done: $out"
done
echo "All done. Re-run run_pareto.py for planes to include tiny results."
