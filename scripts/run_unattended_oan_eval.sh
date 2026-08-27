#!/usr/bin/env bash
# Final link in the Phase 3 chain. Once the OAN arm has trained and scored:
# evaluate it as a cascade, fold it into the law, regenerate the paper's numbers
# and tables, and rebuild the PDF. What it deliberately does NOT do is write the
# prose that replaces the PENDING block in main.tex; deciding how to state a
# result that either confirms or contradicts Xie et al. is not a scripted job.
#
#   setsid nohup scripts/run_unattended_oan_eval.sh > logs/unattended_oan_eval.log 2>&1 </dev/null &
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
cd "$R"
PY="$R/.venv/bin/python"
LOGDIR="$R/logs/oan"; mkdir -p "$LOGDIR"
echo $$ > "$R/logs/unattended_oan_eval.pid"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "OAN-eval runner started (pid $$)"
while pgrep -f "run_unattended_oan.sh|_oan_retry.sh|train_oan.py|train_detector.py" >/dev/null 2>&1; do
  sleep 300
done
log "OAN training stage finished"

SCORES="$R/reports/oan/gate_oan_ships_val.jsonl"
if [ ! -f "$SCORES" ]; then
  log "STOP: no OAN scores at $SCORES; training or scoring did not complete."
  log "      Inspect $LOGDIR/train_oan_ships.log and $LOGDIR/score_oan.log."
  exit 1
fi

if [ -s "$R/runs/oan_last_save_dir.txt" ]; then
  OANDIR=$(cat "$R/runs/oan_last_save_dir.txt")
else
  OANDIR=$(find "$R/runs" -maxdepth 5 -type d -name "oan_ships_lambda3" 2>/dev/null | head -1)
fi
log "OAN run dir resolved to: ${OANDIR:-<none>}"
if [ -z "$OANDIR" ] || [ ! -f "$OANDIR/weights/best.pt" ]; then
  log "STOP: no OAN checkpoint found; training did not produce weights."
  exit 1
fi
PRED="$OANDIR/predict_val"
if [ ! -d "$PRED/labels" ]; then
  log "caching the OAN model's own full-pass predictions"
  "$R/.venv/bin/yolo" predict task=obb model="$OANDIR/weights/best.pt" \
    source="$R/data/processed/dota_ships/images/val" \
    save_txt=True save_conf=True save=False device=1 imgsz=1024 \
    project="$OANDIR" name=predict_val exist_ok=True \
    > "$LOGDIR/predict_oan.log" 2>&1
fi

DET_G=$($PY -c "
import json
print(json.load(open('reports/speed/yolo11m_multiclass_obb_b1.json'))['gflops_per_image'])" 2>/dev/null || echo 184.17)

log "eval_cascade on the fused OAN gate (gate overhead 0: it shares the forward pass)"
$PY scripts/eval_cascade.py \
  --scores "$SCORES" --detector-runs "$PRED" \
  --gt-labels "$R/data/processed/dota_ships/labels/val" \
  --image-root "$R/data/processed/dota_ships/images/val" \
  --all-calibrations \
  --metadata-jsonl "$R/data/processed/dota_ships/metadata/tiles.jsonl" \
  --gate-flops-g 0.0 --detector-flops-g "$DET_G" \
  --label gate_oan_ships --out "$R/reports/oan/cascade_oan_ships.json" \
  > "$LOGDIR/eval_oan.log" 2>&1
log "  exit $?"

log "folding the OAN arm into the law and regenerating paper artifacts"
$PY scripts/savings_model.py  > "$LOGDIR/regen.log" 2>&1
$PY scripts/sparsity_ceiling.py >> "$LOGDIR/regen.log" 2>&1
$PY scripts/paper_numbers.py  >> "$LOGDIR/regen.log" 2>&1
$PY scripts/paper_tables.py   >> "$LOGDIR/regen.log" 2>&1
PYTHONPATH="$R/scripts" $PY scripts/paper_figures_law.py >> "$LOGDIR/regen.log" 2>&1
PYTHONPATH="$R/scripts" $PY scripts/flops_latency_gap.py >> "$LOGDIR/regen.log" 2>&1

log "rebuilding the PDF"
( cd "$R/paper" && cp ../reports/figures/*.pdf figures/ 2>/dev/null;
  PATH="$HOME/bin:$PATH" tectonic -X compile main.tex ) > "$LOGDIR/build.log" 2>&1
log "  build exit $?"

log "=== head-to-head, ships: fused OAN gate vs independent gate ==="
$PY - <<'PY'
import json, glob
S = {d["domain"]: d for d in json.load(open("reports/figures/savings_summary.json"))}
def show(key, label):
    d = S.get(key)
    if not d:
        print(f"  {label}: absent from savings_summary.json"); return
    b = d["by_tolerance_rule"]["absolute"]
    if not b:
        print(f"  {label}: no operating point within tolerance"); return
    print(f"  {label:<34} p+={d['p_plus']:.3f}  saved={100*b['saved_detector_only']:.1f}%"
          f"  TPR={b['tpr']:.3f}  FPR={b['fpr']:.3f}  mAP={b['mAP@0.50']:.3f}")
show("OAN-joint/ships", "fused OAN head (joint)")
show("DOTA-ships", "independent MobileNetV3 gate")
print()
print("  Note: the OAN arm's overhead term is 0 by construction, so its latency-priced")
print("  saving equals its FLOPs-priced saving. The independent gate pays 27.6% at batch 1.")
PY

log ""
log "REMAINING MANUAL STEP: main.tex still contains the PENDING block in the"
log "'What Does Not Move the Frontier' section. Replace it with prose stating the"
log "numbers above, then rebuild. Everything else is regenerated."
log "OAN-eval runner finished"
