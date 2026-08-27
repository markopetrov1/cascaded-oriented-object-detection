#!/usr/bin/env bash
# Multi-class cascade orchestrator. Runs the full per-class sequencer for each
# class in CLASSES, serially. Designed to run unattended in screen so it
# survives terminal disconnect.
#
# Usage::
#   screen -dmS multiclass bash -c 'scripts/run_multiclass.sh 2>&1 | tee -a logs/multiclass.log'
#   screen -r multiclass     # to follow
#
# Env overrides:
#   DEVICE=0
#   CLASSES="ships planes small_vehicle"   # space-separated, in run order
#   DOTA_RAW=data/raw/DOTA
#   SHARED_TILES=data/processed/dota_ships  # already-prepared tile dir; reused
#                                           # by labels-only prep for new classes

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE="${DEVICE:-0}"
# Default skips 'ships' because the existing 'exp' screen handles it.
CLASSES="${CLASSES:-planes small_vehicle}"
DOTA_RAW="${DOTA_RAW:-data/raw/DOTA}"
SHARED_TILES="${SHARED_TILES:-data/processed/dota_ships}"
WAIT_FOR_EXP="${WAIT_FOR_EXP:-1}"   # 1=wait for 'exp' screen to exit before starting

mkdir -p logs

# Map class tag -> DOTA-V2 class string (matching DOTA_V2_CLASSES in src/datasets/converters.py).
declare -A CLASS_TO_DOTA=(
    [ships]="ship"
    [planes]="plane"
    [small_vehicle]="small vehicle"
    [large_vehicle]="large vehicle"
    [storage_tank]="storage tank"
    [harbor]="harbor"
    [bridge]="bridge"
    [helicopter]="helicopter"
    [tennis_court]="tennis court"
    [basketball_court]="basketball court"
    [baseball_diamond]="baseball diamond"
)

prepare_class() {
    local tag="$1"
    local dota_class="${CLASS_TO_DOTA[$tag]:-}"
    if [ -z "$dota_class" ]; then
        echo "ERROR: unknown class tag '$tag' (no entry in CLASS_TO_DOTA)" >&2
        return 2
    fi
    local out_dir="data/processed/dota_${tag}"
    if [ -d "$out_dir/labels/train" ] && [ "$(ls -1 "$out_dir/labels/train" 2>/dev/null | wc -l)" -gt 1000 ]; then
        echo "[multiclass] $tag: data already prepared at $out_dir, skipping prep"
        return 0
    fi
    mkdir -p "$out_dir"
    # Symlink images from the shared tile dir.
    if [ ! -e "$out_dir/images" ]; then
        ln -s "$(realpath "$SHARED_TILES")/images" "$out_dir/images"
    fi
    echo "[multiclass] $tag: regenerating labels (no re-tile) for class '$dota_class' -> $out_dir"
    python3 scripts/prepare_data.py \
        --raw-dir "$DOTA_RAW" \
        --out-dir "$out_dir" \
        --tile-size 1024 --overlap 200 \
        --positive-classes "$dota_class" \
        --class-set v2 \
        --labels-only \
        --no-validate \
        --dataset-yaml "configs/datasets/dota_${tag}.yaml" \
        --stats-out "reports/sparsity_${tag}.json" \
        --skip-split 2>&1 | tee "logs/prepare_${tag}.log"
}

run_class() {
    local tag="$1"
    local sentinel="logs/sentinels/multiclass_${tag}.done"
    if [ -f "$sentinel" ]; then
        echo "[multiclass] $tag: already complete (sentinel exists)"
        return 0
    fi
    echo
    echo "=================================================================="
    echo "[multiclass] BEGIN class=$tag @ $(date '+%F %T')"
    echo "=================================================================="
    prepare_class "$tag"
    PROCESSED="data/processed/dota_${tag}" CLASS_TAG="$tag" DEVICE="$DEVICE" \
        scripts/run_experiment_class.sh
    mkdir -p logs/sentinels
    date > "$sentinel"
    echo "[multiclass] DONE class=$tag @ $(date '+%F %T')"
}

# Wait for any running ship sequencer ('exp' screen) to complete before
# starting class-specific runs. Avoids GPU contention and disk pressure.
if [ "$WAIT_FOR_EXP" = "1" ]; then
    while screen -ls 2>/dev/null | grep -q '\.exp\b'; do
        echo "[multiclass] $(date '+%F %T') waiting for 'exp' screen to finish (ship cascade)..."
        sleep 300
    done
    echo "[multiclass] 'exp' screen has exited; proceeding with multi-class runs."
fi

# --- main loop ---
for tag in $CLASSES; do
    run_class "$tag" || {
        echo "[multiclass] $tag FAILED — continuing with remaining classes"
    }
done

echo
echo "=================================================================="
echo "[multiclass] ALL CLASSES COMPLETE @ $(date '+%F %T')"
echo "=================================================================="

# Final aggregate Pareto across all classes.
python3 -c "
import json, glob
from pathlib import Path
out = {}
for f in glob.glob('reports/*/cascade/*.json'):
    cls = Path(f).parts[1]
    rows = json.loads(Path(f).read_text())
    for r in rows:
        r['class'] = cls
    out.setdefault(cls, []).extend(rows)
agg = [r for rs in out.values() for r in rs]
Path('reports/all_pareto.json').write_text(json.dumps(agg, indent=2))
print(f'aggregated {len(agg)} rows from {len(out)} classes -> reports/all_pareto.json')
"
