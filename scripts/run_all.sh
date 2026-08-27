#!/usr/bin/env bash
# One-command orchestration of the full cascade study.
#
# Usage::
#   scripts/run_all.sh prepare         # tile DOTA + sparsity + geographic split
#   scripts/run_all.sh train_baselines # YOLO26 n/s/m baselines
#   scripts/run_all.sh train_gates     # all 6 gate backbones
#   scripts/run_all.sh score_gates     # produce per-gate score JSONLs
#   scripts/run_all.sh predict_baseline # run YOLO26m predict over all tiles (used by distillation + cascade composition)
#   scripts/run_all.sh codesign        # train shared / early-exit / relaxed
#   scripts/run_all.sh distill         # train detector->gate distilled variants
#   scripts/run_all.sh eval_cascade    # full Pareto sweep (every gate × calibration)
#   scripts/run_all.sh stratified      # re-emit stratified eval against best gate
#   scripts/run_all.sh hrsc            # zero-shot HRSC2016 robustness check
#   scripts/run_all.sh report          # tables + figures (notebook)
#   scripts/run_all.sh full            # everything in order
#
# Environment overrides: DEVICE=0 EPOCHS=30 BATCH=16

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-100}"
GATE_EPOCHS="${GATE_EPOCHS:-20}"
BATCH="${BATCH:-16}"
DOTA_RAW="${DOTA_RAW:-data/raw/DOTA}"
PROCESSED="${PROCESSED:-data/processed/dota_ships}"
HRSC_RAW="${HRSC_RAW:-data/raw/HRSC2016}"

# Cost constants for cascade evaluation. Replace with measured values from
# `python -m src.experiments.speed --weights ... --imgsz 1024 --device $DEVICE`.
GATE_FLOPS_G="${GATE_FLOPS_G:-1.8}"     # ResNet-18 @ 256
GATE_LAT_MS="${GATE_LAT_MS:-2.0}"
DET_FLOPS_G="${DET_FLOPS_G:-91.0}"      # YOLO11m-OBB @ 1024 (rough; replace with measured)
DET_LAT_MS="${DET_LAT_MS:-22.0}"

GATE_NAMES=(gate_resnet18 gate_resnet50 gate_mbv3small gate_mbv3large gate_effb0 gate_tiny)
BASELINE_NAMES=(baseline_yolo11n_obb baseline_yolo11s_obb baseline_yolo11m_obb)

cmd_prepare() {
    python scripts/prepare_data.py \
        --raw-dir "$DOTA_RAW" \
        --out-dir "$PROCESSED" \
        --tile-size 1024 --overlap 200 \
        --positive-classes ship \
        --dataset-yaml configs/datasets/dota_ships.yaml \
        --stats-out reports/sparsity_ships.json \
        --splits-out configs/splits/geographic_v1.yaml
}

cmd_train_baselines() {
    for name in "${BASELINE_NAMES[@]}"; do
        python scripts/train_detector.py \
            --config "configs/experiments/${name}.yaml" \
            --device "$DEVICE" --epochs "$EPOCHS"
    done
}

cmd_train_gates() {
    for name in "${GATE_NAMES[@]}"; do
        python scripts/train_gate.py \
            --config "configs/experiments/${name}.yaml" \
            --device "$DEVICE" --epochs "$GATE_EPOCHS"
    done
}

cmd_score_gates() {
    mkdir -p reports/gate_scores
    for name in "${GATE_NAMES[@]}"; do
        for split in val test; do
            [ -f "runs/${name}/best.pt" ] || { echo "skip ${name}: no checkpoint"; continue; }
            python scripts/score_tiles.py \
                --weights "runs/${name}/best.pt" \
                --data-root "$PROCESSED" --split "$split" \
                --out "reports/gate_scores/${name}_${split}.jsonl" \
                --device "$DEVICE"
        done
    done
}

cmd_predict_baseline() {
    # Ultralytics CLI is the simplest path. The save_txt + save_conf flags
    # produce the format `cascade.load_detector_outputs_yolo_obb` consumes.
    for split in val test; do
        yolo predict \
            task=obb \
            model=runs/baseline_yolo11m_dota_ships_obb/weights/best.pt \
            source="$PROCESSED/images/${split}" \
            save_txt=True save_conf=True \
            project=runs/baseline_yolo11m_dota_ships_obb \
            name="predict_${split}" exist_ok=True
    done
}

cmd_codesign() {
    for variant in shared early_exit relaxed; do
        python scripts/train_codesign.py \
            --variant "$variant" \
            --backbone resnet18 \
            --data-root "$PROCESSED" \
            --epochs 30 --device "cuda:${DEVICE}" \
            --name "codesign_${variant}_resnet18"
    done
}

cmd_distill() {
    for name in "${GATE_NAMES[@]}"; do
        python scripts/train_distill.py \
            --gate-config "configs/experiments/${name}.yaml" \
            --detector-runs runs/baseline_yolo11m_dota_ships_obb/predict_train \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" \
            --device "$DEVICE" --name "${name}_distill"
    done
}

cmd_eval_cascade() {
    mkdir -p reports/cascade
    # Oracle baseline.
    python scripts/eval_cascade.py \
        --gate oracle \
        --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
        --detector-runs runs/baseline_yolo11m_dota_ships_obb/predict_val \
        --gt-labels "$PROCESSED/labels/val" \
        --image-root "$PROCESSED/images/val" \
        --gate-flops-g 0 --gate-latency-ms 0 \
        --detector-flops-g "$DET_FLOPS_G" --detector-latency-ms "$DET_LAT_MS" \
        --label oracle \
        --out reports/cascade/oracle.json

    # Real gates × {identity, temperature, platt, isotonic}.
    for name in "${GATE_NAMES[@]}"; do
        for calib in identity temperature platt isotonic; do
            scores="reports/gate_scores/${name}_val.jsonl"
            [ -f "$scores" ] || { echo "skip ${name}/${calib}: missing scores"; continue; }
            python scripts/eval_cascade.py \
                --scores "$scores" \
                --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
                --detector-runs runs/baseline_yolo11m_dota_ships_obb/predict_val \
                --gt-labels "$PROCESSED/labels/val" \
                --image-root "$PROCESSED/images/val" \
                --calibration "$calib" \
                --gate-flops-g "$GATE_FLOPS_G" --gate-latency-ms "$GATE_LAT_MS" \
                --detector-flops-g "$DET_FLOPS_G" --detector-latency-ms "$DET_LAT_MS" \
                --label "${name}" \
                --out "reports/cascade/${name}_${calib}.json"
        done
    done
}

cmd_stratified() {
    # Pick the best gate (override BEST_GATE env to change).
    BEST_GATE="${BEST_GATE:-gate_resnet18}"
    for stratum in size_bucket gsd_bucket imagesource boundary; do
        python scripts/eval_cascade.py \
            --scores "reports/gate_scores/${BEST_GATE}_val.jsonl" \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
            --detector-runs runs/baseline_yolo11m_dota_ships_obb/predict_val \
            --gt-labels "$PROCESSED/labels/val" \
            --image-root "$PROCESSED/images/val" \
            --calibration temperature \
            --gate-flops-g "$GATE_FLOPS_G" --gate-latency-ms "$GATE_LAT_MS" \
            --detector-flops-g "$DET_FLOPS_G" --detector-latency-ms "$DET_LAT_MS" \
            --label "${BEST_GATE}_temperature" \
            --stratum "$stratum" \
            --out "reports/cascade/${BEST_GATE}_temperature.json"
    done
}

cmd_hrsc() {
    python scripts/prepare_hrsc.py --raw-dir "$HRSC_RAW" --out-dir data/processed/hrsc2016 --splits test
    yolo predict \
        task=obb \
        model=runs/baseline_yolo11m_dota_ships_obb/weights/best.pt \
        source=data/processed/hrsc2016/images/test \
        save_txt=True save_conf=True \
        project=runs/baseline_yolo11m_dota_ships_obb \
        name=predict_hrsc_test exist_ok=True
    BEST_GATE="${BEST_GATE:-gate_resnet18}"
    python scripts/score_tiles.py \
        --weights "runs/${BEST_GATE}/best.pt" \
        --data-root data/processed/hrsc2016 --split test \
        --out "reports/gate_scores/${BEST_GATE}_hrsc_test.jsonl" \
        --device "$DEVICE"
    python scripts/eval_cascade.py \
        --scores "reports/gate_scores/${BEST_GATE}_hrsc_test.jsonl" \
        --metadata-jsonl data/processed/hrsc2016/metadata/tiles.jsonl --split test \
        --detector-runs runs/baseline_yolo11m_dota_ships_obb/predict_hrsc_test \
        --gt-labels data/processed/hrsc2016/labels/test \
        --image-root data/processed/hrsc2016/images/test \
        --calibration temperature \
        --gate-flops-g "$GATE_FLOPS_G" --gate-latency-ms "$GATE_LAT_MS" \
        --detector-flops-g "$DET_FLOPS_G" --detector-latency-ms "$DET_LAT_MS" \
        --label hrsc_zero_shot \
        --out reports/cascade/hrsc2016_zero_shot.json
}

cmd_report() {
    python scripts/run_pareto.py \
        --inputs reports/cascade/*.json \
        --out reports/pareto.csv \
        --figure reports/figures/pareto
    if command -v jupyter >/dev/null 2>&1; then
        jupyter nbconvert --to notebook --execute notebooks/01_master_pareto.ipynb \
            --output 01_master_pareto.executed.ipynb || true
    fi
}

cmd_full() {
    cmd_prepare
    cmd_train_baselines
    cmd_train_gates
    cmd_score_gates
    cmd_predict_baseline
    cmd_codesign
    cmd_distill
    cmd_eval_cascade
    cmd_stratified
    cmd_hrsc
    cmd_report
}

target="${1:-help}"
case "$target" in
    prepare) cmd_prepare ;;
    train_baselines) cmd_train_baselines ;;
    train_gates) cmd_train_gates ;;
    score_gates) cmd_score_gates ;;
    predict_baseline) cmd_predict_baseline ;;
    codesign) cmd_codesign ;;
    distill) cmd_distill ;;
    eval_cascade) cmd_eval_cascade ;;
    stratified) cmd_stratified ;;
    hrsc) cmd_hrsc ;;
    report) cmd_report ;;
    full) cmd_full ;;
    *)
        echo "Usage: $0 {prepare|train_baselines|train_gates|score_gates|predict_baseline|codesign|distill|eval_cascade|stratified|hrsc|report|full}"
        exit 1
        ;;
esac
