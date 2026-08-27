#!/usr/bin/env bash
# Full experiment sequencer. Designed to be invoked from a screen session and
# run unattended through all phases. Each step writes a sentinel under
# logs/sentinels/ on success so reruns skip what's already done.
#
# Usage::
#   screen -dmS exp bash -c 'scripts/run_experiment.sh 2>&1 | tee -a logs/exp.log'
#   screen -r exp        # to follow
#
# Env overrides (defaults shown):
#   DEVICE=0
#   DET_EPOCHS=80              # YOLO baselines
#   GATE_EPOCHS=20             # gate backbones
#   CODESIGN_EPOCHS=25         # phase 3 variants
#   SMOKE_DET_EPOCHS=1
#   SMOKE_GATE_EPOCHS=1
#   IMG_SIZE=1024              # detector tile size
#   GATE_IMG_SIZE=256          # gate input size

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE="${DEVICE:-0}"
DET_EPOCHS="${DET_EPOCHS:-50}"
GATE_EPOCHS="${GATE_EPOCHS:-15}"
CODESIGN_EPOCHS="${CODESIGN_EPOCHS:-20}"
SMOKE_DET_EPOCHS="${SMOKE_DET_EPOCHS:-1}"
SMOKE_GATE_EPOCHS="${SMOKE_GATE_EPOCHS:-1}"
IMG_SIZE="${IMG_SIZE:-1024}"
GATE_IMG_SIZE="${GATE_IMG_SIZE:-256}"
PROCESSED="${PROCESSED:-data/processed/dota_ships}"

mkdir -p logs/sentinels

run_step() {
    local name="$1"
    shift
    local sentinel="logs/sentinels/${name}.done"
    if [ -f "$sentinel" ]; then
        echo "[exp] SKIP $name (sentinel exists: $sentinel)"
        return 0
    fi
    echo
    echo "================================================================"
    echo "[exp] START $name @ $(date '+%F %T')"
    echo "================================================================"
    if "$@"; then
        date > "$sentinel"
        echo "[exp] DONE  $name @ $(date '+%F %T')"
    else
        echo "[exp] FAIL  $name (exit $?). Stopping."
        return 1
    fi
}

step_validate() {
    python3 scripts/validate_prep.py "$PROCESSED"
}

step_smoke_detector() {
    python3 scripts/train_detector.py \
        --config configs/experiments/baseline_yolo11n_obb.yaml \
        --device "$DEVICE" --epochs "$SMOKE_DET_EPOCHS" \
        --name baseline_yolo11n_smoke
}

step_train_yolo11n() {
    python3 scripts/train_detector.py \
        --config configs/experiments/baseline_yolo11n_obb.yaml \
        --device "$DEVICE" --epochs "$DET_EPOCHS"
}

step_train_yolo11s() {
    python3 scripts/train_detector.py \
        --config configs/experiments/baseline_yolo11s_obb.yaml \
        --device "$DEVICE" --epochs "$DET_EPOCHS"
}

step_train_yolo11m() {
    python3 scripts/train_detector.py \
        --config configs/experiments/baseline_yolo11m_obb.yaml \
        --device "$DEVICE" --epochs "$DET_EPOCHS"
}

step_smoke_gate() {
    python3 scripts/train_gate.py \
        --config configs/experiments/gate_resnet18.yaml \
        --device "$DEVICE" --epochs "$SMOKE_GATE_EPOCHS" \
        --name gate_resnet18_smoke
}

step_train_gates() {
    for cfg in configs/experiments/gate_*.yaml; do
        name=$(basename "$cfg" .yaml)
        if [ -f "logs/sentinels/gate_${name}.done" ]; then
            echo "[exp] SKIP gate $name"
            continue
        fi
        python3 scripts/train_gate.py \
            --config "$cfg" --device "$DEVICE" --epochs "$GATE_EPOCHS"
        date > "logs/sentinels/gate_${name}.done"
    done
}

step_predict_baseline() {
    local best="runs/obb/runs/baseline_yolo11m_dota_ships_obb/weights/best.pt"
    if [ ! -f "$best" ]; then
        echo "[exp] missing $best; skip"
        return 1
    fi
    for split in val test; do
        out_dir="runs/obb/runs/baseline_yolo11m_dota_ships_obb/predict_${split}"
        if [ -d "$out_dir/labels" ] && [ "$(ls -1 "$out_dir/labels" 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "[exp] predict_${split} already populated; skip"
            continue
        fi
        yolo predict task=obb model="$best" \
            source="$PROCESSED/images/${split}" \
            save_txt=True save_conf=True \
            project=runs/predictions \
            name="baseline_yolo11m_dota_ships_obb_${split}" exist_ok=True \
            device="$DEVICE" verbose=False
        actual=""
        for cand in "runs/obb/runs/predictions/baseline_yolo11m_dota_ships_obb_${split}" "runs/predictions/baseline_yolo11m_dota_ships_obb_${split}"; do
            if [ -d "$cand/labels" ]; then actual="$cand"; break; fi
        done
        [ -n "$actual" ] && ln -sfn "$(realpath "$actual")" "$out_dir"
    done
}

step_score_gates() {
    mkdir -p reports/gate_scores
    for ckpt in runs/gate_*/best.pt; do
        name=$(basename "$(dirname "$ckpt")")
        for split in val test; do
            out="reports/gate_scores/${name}_${split}.jsonl"
            if [ -f "$out" ] && [ "$(wc -l < "$out")" -gt 0 ]; then
                echo "[exp] $out exists; skip"
                continue
            fi
            python3 scripts/score_tiles.py \
                --weights "$ckpt" \
                --data-root "$PROCESSED" --split "$split" \
                --out "$out" --device "$DEVICE"
        done
    done
}

step_eval_cascade() {
    mkdir -p reports/cascade
    # Oracle baseline.
    if [ ! -f reports/cascade/oracle.json ]; then
        python3 scripts/eval_cascade.py \
            --gate oracle \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
            --detector-runs runs/obb/runs/baseline_yolo11m_dota_ships_obb/predict_val \
            --gt-labels "$PROCESSED/labels/val" \
            --image-root "$PROCESSED/images/val" \
            --gate-flops-g 0 --gate-latency-ms 0 \
            --detector-flops-g 91 --detector-latency-ms 22 \
            --label oracle \
            --out reports/cascade/oracle.json
    fi
    for ckpt in runs/gate_*/best.pt; do
        name=$(basename "$(dirname "$ckpt")")
        scores="reports/gate_scores/${name}_val.jsonl"
        [ -f "$scores" ] || { echo "skip $name: no scores"; continue; }
        # Check if all 4 calibration outputs exist — skip the whole gate if so.
        all_done=1
        for calib in identity temperature platt isotonic; do
            [ -f "reports/cascade/${name}_${calib}.json" ] || { all_done=0; break; }
        done
        if [ "$all_done" = "1" ]; then
            echo "[exp] all calibrations for $name exist; skip"
            continue
        fi
        # Run all calibrations in one pass (precompute IoU once per gate).
        python3 scripts/eval_cascade.py \
            --scores "$scores" \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
            --detector-runs runs/obb/runs/baseline_yolo11m_dota_ships_obb/predict_val \
            --gt-labels "$PROCESSED/labels/val" \
            --image-root "$PROCESSED/images/val" \
            --all-calibrations \
            --gate-flops-g 1.8 --gate-latency-ms 2.0 \
            --detector-flops-g 91 --detector-latency-ms 22 \
            --label "$name" \
            --out "reports/cascade/${name}_identity.json"
    done
}

step_codesign() {
    for variant in shared early_exit relaxed; do
        name="codesign_${variant}_resnet18"
        sentinel="logs/sentinels/${name}.done"
        [ -f "$sentinel" ] && { echo "skip $name"; continue; }
        python3 scripts/train_codesign.py \
            --variant "$variant" --backbone resnet18 \
            --data-root "$PROCESSED" \
            --epochs "$CODESIGN_EPOCHS" --device "cuda:${DEVICE}" \
            --name "$name" \
            --image-size "$GATE_IMG_SIZE" \
            --batch-size 32 || echo "[exp] $variant failed; recording and continuing"
        date > "$sentinel"
    done
}

step_distill() {
    for cfg in configs/experiments/gate_*.yaml; do
        name=$(basename "$cfg" .yaml)_distill
        sentinel="logs/sentinels/${name}.done"
        [ -f "$sentinel" ] && { echo "skip $name"; continue; }
        python3 scripts/train_distill.py \
            --gate-config "$cfg" \
            --detector-runs runs/obb/runs/baseline_yolo11m_dota_ships_obb/predict_val \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" \
            --device "$DEVICE" --name "$name" \
            --epochs "$GATE_EPOCHS" || echo "[exp] $name failed; continuing"
        date > "$sentinel"
    done
}

step_stratified() {
    BEST_GATE="${BEST_GATE:-gate_resnet18}"
    scores="reports/gate_scores/${BEST_GATE}_val.jsonl"
    [ -f "$scores" ] || { echo "no scores for $BEST_GATE"; return 1; }
    for stratum in size_bucket gsd_bucket imagesource boundary; do
        out="reports/cascade/${BEST_GATE}_temperature.json"
        # eval_cascade emits sibling stratified file alongside the main JSON.
        python3 scripts/eval_cascade.py \
            --scores "$scores" \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
            --detector-runs runs/obb/runs/baseline_yolo11m_dota_ships_obb/predict_val \
            --gt-labels "$PROCESSED/labels/val" \
            --image-root "$PROCESSED/images/val" \
            --calibration temperature \
            --gate-flops-g 1.8 --gate-latency-ms 2.0 \
            --detector-flops-g 91 --detector-latency-ms 22 \
            --label "${BEST_GATE}_temperature" \
            --stratum "$stratum" \
            --out "$out"
    done
}

step_report() {
    python3 scripts/run_pareto.py \
        --inputs reports/cascade/*.json \
        --out reports/pareto.csv \
        --figure reports/figures/pareto
}

# --- main sequence ---

run_step validate            step_validate
run_step smoke_detector      step_smoke_detector
run_step train_yolo11n       step_train_yolo11n
run_step train_yolo11s       step_train_yolo11s
run_step train_yolo11m       step_train_yolo11m
run_step smoke_gate          step_smoke_gate
run_step train_gates         step_train_gates
run_step predict_baseline    step_predict_baseline
run_step score_gates         step_score_gates
run_step eval_cascade        step_eval_cascade
run_step stratified          step_stratified
run_step codesign            step_codesign
run_step distill             step_distill
run_step report              step_report

echo
echo "================================================================"
echo "[exp] ALL STEPS COMPLETE @ $(date '+%F %T')"
echo "================================================================"
