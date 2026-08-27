#!/usr/bin/env bash
# Per-class cascade experiment sequencer. Same pipeline as run_experiment.sh
# but parameterized by CLASS_TAG (e.g. "ships", "planes", "small_vehicle").
#
# Required env:
#   CLASS_TAG      — short tag used in run names + dir paths (no spaces)
#   PROCESSED      — path to data/processed/dota_<tag> directory
#
# Optional env (same defaults as run_experiment.sh):
#   DEVICE=0
#   DET_EPOCHS=50, GATE_EPOCHS=15, CODESIGN_EPOCHS=20
#   IMG_SIZE=1024, GATE_IMG_SIZE=256
#
# Each step's stdout+stderr goes to logs/<tag>/<step>.log AND the global tee.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${CLASS_TAG:-}" ]; then
    echo "ERROR: CLASS_TAG env var required (e.g. ships, planes, small_vehicle)" >&2
    exit 2
fi
if [ -z "${PROCESSED:-}" ]; then
    PROCESSED="data/processed/dota_${CLASS_TAG}"
fi
if [ ! -d "$PROCESSED" ]; then
    echo "ERROR: processed dir $PROCESSED not found" >&2
    exit 2
fi

DEVICE="${DEVICE:-0}"
DET_EPOCHS="${DET_EPOCHS:-50}"
GATE_EPOCHS="${GATE_EPOCHS:-15}"
CODESIGN_EPOCHS="${CODESIGN_EPOCHS:-20}"
IMG_SIZE="${IMG_SIZE:-1024}"
GATE_IMG_SIZE="${GATE_IMG_SIZE:-256}"

LOG_DIR="logs/${CLASS_TAG}"
SENT_DIR="logs/sentinels/${CLASS_TAG}"
mkdir -p "$LOG_DIR" "$SENT_DIR"

# Generate per-class configs by sed-replacing 'ships' tag in the original ship configs.
GENERATED_CFG_DIR="configs/experiments/_generated/${CLASS_TAG}"
mkdir -p "$GENERATED_CFG_DIR"
for src in configs/experiments/baseline_yolo11n_obb.yaml \
           configs/experiments/baseline_yolo11s_obb.yaml \
           configs/experiments/baseline_yolo11m_obb.yaml \
           configs/experiments/gate_resnet18.yaml \
           configs/experiments/gate_resnet50.yaml \
           configs/experiments/gate_mbv3small.yaml \
           configs/experiments/gate_mbv3large.yaml \
           configs/experiments/gate_effb0.yaml \
           configs/experiments/gate_tiny.yaml; do
    base=$(basename "$src" .yaml)
    out="$GENERATED_CFG_DIR/${base}.yaml"
    sed -e "s|dota_ships|dota_${CLASS_TAG}|g" \
        -e "s|_dota_ships_|_dota_${CLASS_TAG}_|g" \
        -e "s|name: gate_|name: gate_${CLASS_TAG}_|g" \
        "$src" > "$out"
done

run_step() {
    local name="$1"
    shift
    local sentinel="${SENT_DIR}/${name}.done"
    local log_path="${LOG_DIR}/${name}.log"
    if [ -f "$sentinel" ]; then
        echo "[exp:${CLASS_TAG}] SKIP $name (sentinel: $sentinel)"
        return 0
    fi
    echo
    echo "================================================================"
    echo "[exp:${CLASS_TAG}] START $name @ $(date '+%F %T')  (log: $log_path)"
    echo "================================================================"
    if "$@" 2>&1 | tee "$log_path"; then
        date > "$sentinel"
        echo "[exp:${CLASS_TAG}] DONE  $name @ $(date '+%F %T')"
    else
        # The pipe-tee makes "$@" status fall away; check tee exit instead.
        local tee_exit=${PIPESTATUS[0]}
        if [ "$tee_exit" = 0 ]; then
            date > "$sentinel"
            echo "[exp:${CLASS_TAG}] DONE  $name @ $(date '+%F %T')"
        else
            echo "[exp:${CLASS_TAG}] FAIL  $name (cmd exit $tee_exit). Stopping."
            return 1
        fi
    fi
}

# Resolve detector model name with class suffix.
DET_NAME_M="baseline_yolo11m_dota_${CLASS_TAG}_obb"
DET_RUN_DIR="runs/obb/runs/${DET_NAME_M}"

step_validate() {
    python3 scripts/validate_prep.py "$PROCESSED"
}

step_train_yolo11n() {
    python3 scripts/train_detector.py \
        --config "$GENERATED_CFG_DIR/baseline_yolo11n_obb.yaml" \
        --device "$DEVICE" --epochs "$DET_EPOCHS"
}
step_train_yolo11s() {
    python3 scripts/train_detector.py \
        --config "$GENERATED_CFG_DIR/baseline_yolo11s_obb.yaml" \
        --device "$DEVICE" --epochs "$DET_EPOCHS"
}
step_train_yolo11m() {
    python3 scripts/train_detector.py \
        --config "$GENERATED_CFG_DIR/baseline_yolo11m_obb.yaml" \
        --device "$DEVICE" --epochs "$DET_EPOCHS"
}

step_train_gates() {
    for cfg in "$GENERATED_CFG_DIR"/gate_*.yaml; do
        name=$(basename "$cfg" .yaml)
        if [ -f "${SENT_DIR}/gate_${name}.done" ]; then
            echo "[exp:${CLASS_TAG}] SKIP gate $name"
            continue
        fi
        python3 scripts/train_gate.py \
            --config "$cfg" --device "$DEVICE" --epochs "$GATE_EPOCHS" 2>&1 | tee -a "${LOG_DIR}/gate_${name}.log"
        date > "${SENT_DIR}/gate_${name}.done"
    done
}

step_predict_baseline() {
    local best="${DET_RUN_DIR}/weights/best.pt"
    if [ ! -f "$best" ]; then
        echo "[exp:${CLASS_TAG}] missing $best; skip"
        return 1
    fi
    for split in val test; do
        out_dir="${DET_RUN_DIR}/predict_${split}"
        if [ -d "$out_dir/labels" ] && [ "$(ls -1 "$out_dir/labels" 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "[exp:${CLASS_TAG}] predict_${split} already populated; skip"
            continue
        fi
        yolo predict task=obb model="$best" \
            source="$PROCESSED/images/${split}" \
            save_txt=True save_conf=True \
            project=runs/predictions \
            name="${DET_NAME_M}_${split}" exist_ok=True \
            device="$DEVICE" verbose=False
        # Ultralytics prepends `obb/runs/` for OBB task even when project is set explicitly,
        # so resolve the actual save dir (it could be `runs/predictions/X` or `runs/obb/runs/predictions/X`).
        actual=""
        for cand in "runs/obb/runs/predictions/${DET_NAME_M}_${split}" "runs/predictions/${DET_NAME_M}_${split}"; do
            if [ -d "$cand/labels" ]; then actual="$cand"; break; fi
        done
        if [ -z "$actual" ]; then
            echo "[exp:${CLASS_TAG}] could not locate predict output for ${split}"
            return 1
        fi
        ln -sfn "$(realpath "$actual")" "$out_dir"
    done
}

step_score_gates() {
    mkdir -p "reports/${CLASS_TAG}/gate_scores"
    for ckpt in runs/gate_${CLASS_TAG}_*/best.pt; do
        [ -f "$ckpt" ] || { echo "no gate ckpts for ${CLASS_TAG}"; continue; }
        name=$(basename "$(dirname "$ckpt")")
        for split in val test; do
            out="reports/${CLASS_TAG}/gate_scores/${name}_${split}.jsonl"
            if [ -f "$out" ] && [ "$(wc -l < "$out")" -gt 0 ]; then
                echo "[exp:${CLASS_TAG}] $out exists; skip"
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
    local cascade_dir="reports/${CLASS_TAG}/cascade"
    mkdir -p "$cascade_dir"
    if [ ! -f "$cascade_dir/oracle.json" ]; then
        python3 scripts/eval_cascade.py \
            --gate oracle \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
            --detector-runs "${DET_RUN_DIR}/predict_val" \
            --gt-labels "$PROCESSED/labels/val" \
            --image-root "$PROCESSED/images/val" \
            --gate-flops-g 0 --gate-latency-ms 0 \
            --detector-flops-g 91 --detector-latency-ms 22 \
            --label "oracle_${CLASS_TAG}" \
            --out "$cascade_dir/oracle.json"
    fi
    for ckpt in runs/gate_${CLASS_TAG}_*/best.pt; do
        [ -f "$ckpt" ] || continue
        name=$(basename "$(dirname "$ckpt")")
        scores="reports/${CLASS_TAG}/gate_scores/${name}_val.jsonl"
        [ -f "$scores" ] || { echo "skip $name: no scores"; continue; }
        # Check if all 4 calibration outputs exist — skip the whole gate if so.
        all_done=1
        for calib in identity temperature platt isotonic; do
            [ -f "$cascade_dir/${name}_${calib}.json" ] || { all_done=0; break; }
        done
        if [ "$all_done" = "1" ]; then
            echo "[exp:${CLASS_TAG}] all calibrations for $name exist; skip"
            continue
        fi
        # Run all calibrations in one pass (precompute IoU once per gate).
        python3 scripts/eval_cascade.py \
            --scores "$scores" \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
            --detector-runs "${DET_RUN_DIR}/predict_val" \
            --gt-labels "$PROCESSED/labels/val" \
            --image-root "$PROCESSED/images/val" \
            --all-calibrations \
            --gate-flops-g 1.8 --gate-latency-ms 2.0 \
            --detector-flops-g 91 --detector-latency-ms 22 \
            --label "${name}" \
            --out "$cascade_dir/${name}_identity.json"
    done
}

step_codesign() {
    for variant in shared early_exit relaxed; do
        name="codesign_${CLASS_TAG}_${variant}_resnet18"
        sentinel="${SENT_DIR}/${name}.done"
        [ -f "$sentinel" ] && { echo "skip $name"; continue; }
        python3 scripts/train_codesign.py \
            --variant "$variant" --backbone resnet18 \
            --data-root "$PROCESSED" \
            --epochs "$CODESIGN_EPOCHS" --device "cuda:${DEVICE}" \
            --name "$name" \
            --image-size "$GATE_IMG_SIZE" \
            --batch-size 32 2>&1 | tee "${LOG_DIR}/${name}.log" \
            || echo "[exp:${CLASS_TAG}] $variant failed; recording and continuing"
        date > "$sentinel"
    done
}

step_distill() {
    for cfg in "$GENERATED_CFG_DIR"/gate_*.yaml; do
        base=$(basename "$cfg" .yaml)
        name="${base}_distill"
        sentinel="${SENT_DIR}/${name}.done"
        [ -f "$sentinel" ] && { echo "skip $name"; continue; }
        python3 scripts/train_distill.py \
            --gate-config "$cfg" \
            --detector-runs "${DET_RUN_DIR}/predict_val" \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" \
            --device "$DEVICE" --name "$name" \
            --epochs "$GATE_EPOCHS" 2>&1 | tee "${LOG_DIR}/${name}.log" \
            || echo "[exp:${CLASS_TAG}] $name failed; continuing"
        date > "$sentinel"
    done
}

step_stratified() {
    BEST_GATE="${BEST_GATE:-gate_${CLASS_TAG}_resnet18}"
    scores="reports/${CLASS_TAG}/gate_scores/${BEST_GATE}_val.jsonl"
    [ -f "$scores" ] || { echo "no scores for $BEST_GATE"; return 0; }
    local cascade_dir="reports/${CLASS_TAG}/cascade"
    mkdir -p "$cascade_dir"
    for stratum in size_bucket gsd_bucket imagesource boundary; do
        out="$cascade_dir/${BEST_GATE}_temperature_strat_${stratum}.json"
        python3 scripts/eval_cascade.py \
            --scores "$scores" \
            --metadata-jsonl "$PROCESSED/metadata/tiles.jsonl" --split val \
            --detector-runs "${DET_RUN_DIR}/predict_val" \
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
        --inputs "reports/${CLASS_TAG}/cascade"/*.json \
        --out "reports/${CLASS_TAG}/pareto.csv" \
        --figure "reports/${CLASS_TAG}/figures/pareto" || true
}

# --- main sequence ---

run_step validate           step_validate
run_step train_yolo11n      step_train_yolo11n
run_step train_yolo11s      step_train_yolo11s
run_step train_yolo11m      step_train_yolo11m
run_step train_gates        step_train_gates
run_step predict_baseline   step_predict_baseline
run_step score_gates        step_score_gates
run_step eval_cascade       step_eval_cascade
run_step stratified         step_stratified
run_step codesign           step_codesign
run_step distill            step_distill
run_step report             step_report

echo
echo "================================================================"
echo "[exp:${CLASS_TAG}] ALL STEPS COMPLETE @ $(date '+%F %T')"
echo "================================================================"
