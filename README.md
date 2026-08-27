# Cascaded RS Detection

Cascaded object detection for remote-sensing imagery: a tile-level binary gate
followed by an oriented-bounding-box detector. The contribution is the
*systematic study* of when, how, and why cascading helps in RS — see
[PROJECT_PLAN.md](PROJECT_PLAN.md) for the full research plan and
[~/.claude/plans/here-i-have-a-starry-sunbeam.md](../.claude/plans/here-i-have-a-starry-sunbeam.md)
for the phased implementation plan.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hardware: pinned to `cuda4` (`Quadro RTX 8000`, 48GB, Turing → FP16 AMP). Driver
CUDA 12.6.

## One-shot pipeline

```bash
DEVICE=0 EPOCHS=100 GATE_EPOCHS=20 BATCH=16 scripts/run_all.sh full
```

Or one stage at a time — see `scripts/run_all.sh` for the menu.

## Data preparation

DOTA must be downloaded separately and placed at `data/raw/DOTA/{train,val,test}/`
with `images/` and `labelTxt/` subdirs (DOTA's standard layout).

```bash
scripts/run_all.sh prepare
# or, manually:
python scripts/prepare_data.py \
  --raw-dir data/raw/DOTA \
  --out-dir data/processed/dota_ships \
  --tile-size 1024 --overlap 200 \
  --positive-classes ship \
  --dataset-yaml configs/datasets/dota_ships.yaml \
  --stats-out reports/sparsity_ships.json \
  --splits-out configs/splits/geographic_v1.yaml
```

Outputs:

- `data/processed/dota_ships/images/{train,val,test}/<tile>.png`
- `data/processed/dota_ships/labels/{train,val,test}/<tile>.txt` — YOLO-OBB labels (ship-only when `--positive-classes ship`)
- `data/processed/dota_ships/gate_labels/{train,val,test}/<tile>.txt` — binary `0`/`1`
- `data/processed/dota_ships/metadata/tiles.jsonl` — one row per tile with `imagesource`, `gsd`, `is_positive`, ...
- `reports/sparsity_ships.json` (+ `.csv`) — per-class / per-imagesource / per-split positive rates and distributions
- `configs/splits/geographic_v1.yaml` — stem-disjoint, imagesource-stratified train/val/test partition

## Pillar 1 — Pareto / Economics

```bash
scripts/run_all.sh train_baselines       # YOLO11 n/s/m  (note: the trained models are YOLO11-OBB; YOLO26 was the
                                         #                original plan but Ultralytics 8.4.x ships YOLO11)
scripts/run_all.sh train_gates           # 6 gate backbones (resnet18/50, mbv3 s/l, effb0, tiny)
scripts/run_all.sh score_gates           # produce per-gate score JSONLs
scripts/run_all.sh predict_baseline      # cache YOLO11m predictions over all tiles
scripts/run_all.sh eval_cascade          # full Pareto sweep incl. oracle baseline
```

The cascade composition is *cached*: `eval_cascade` reads tile scores +
already-run detector outputs and re-applies thresholds in a tight loop. One
detector pass over the tiles is enough for hundreds of Pareto points.

`eval_cascade.py` now supports `--all-calibrations`: precompute the IoU
matrices once per gate, then sweep all four calibrations in a single process
(4× speedup vs running each calibration separately). It also reports
`mAP@0.5:0.95` as the mean over a 10-point IoU sweep `(0.50, 0.55, …, 0.95)`,
not the previous degenerate `(0.50,)` single-IoU value.

## Pillar 2 — Calibration

Methods 1–4 (naive, recall-targeted, mAP-grid, temperature/Platt/isotonic) live
in `src/calibration.py`. Method 5 (context-adaptive) and Method 6 (learned
threshold MLP) consume cheap per-tile features:

```bash
python scripts/extract_tile_features.py --data-root data/processed/dota_ships --split val \
    --out reports/tile_features/val.jsonl
```

`eval_cascade.py --calibration {temperature|platt|isotonic}` fits the calibrator
on the supplied scores (or `--calibration-fit-scores`) and threshold-sweeps the
calibrated probs.

## Pillar 3 — Co-design

```bash
scripts/run_all.sh codesign      # shared / early-exit / relaxed (Gumbel-softmax)
scripts/run_all.sh distill       # detector → gate distillation
```

`src/codesign.py` provides:
- `SharedBackboneCascade` — one backbone, two heads
- `EarlyExitWrapper` — binary head on an early backbone block, conditional continuation
- `RelaxedGatingCascade` — Gumbel-softmax + straight-through estimator. `forward()` delegates to `forward_train()` so the module is callable from the standard training loop.
- `soft_target_from_detector` + `distillation_loss` — detector → gate distillation

`scripts/train_codesign.py` uses a **gate-warmup loss schedule** by default
to avoid the failure mode where the dense per-cell BCE in `SimpleOBBHead`
corrupts the shared-backbone features the gate task needs:

- Epochs 1–`--gate-warmup-epochs` (default 5): `obb_weight = 0` (gate-only)
- Next 3 epochs: linear ramp from 0 → `--obb-weight` (default `0.1`)
- Remaining epochs: full `--obb-weight`

Set `--gate-warmup-epochs 0 --obb-weight 1.0` to reproduce the original
joint-loss-from-epoch-1 behavior; without the warmup the joint loss collapses
gate PR-AUC to 0.10–0.30 across all backbones (documented negative result).

## Pillar 4 (lens) — RS specifics

```bash
scripts/run_all.sh stratified  # re-run eval, stratified by gsd/size/imagesource/boundary
```

Each stratum run writes a sibling `<out>.stratified_<stratum>.{json,csv}` file
with one row per (threshold, bucket). Each row carries `mAP@0.50` … `mAP@0.95`,
`mAP@0.5:0.95`, plus the per-bucket `filter_rate` and
`gate_recall_on_positive_tiles` — enough to reconstruct a Pareto curve inside
each bucket, not just the per-bucket mAP at one fixed gate threshold.

## Speed / FLOPs benchmarking

`scripts/benchmark_speed.py` measures real GFLOPs and per-image latency for
the trained detector and gate models on the target hardware (Quadro RTX 8000),
replacing the previously hardcoded `--gate-flops-g 1.8 --detector-flops-g 91`
constants used by `eval_cascade.py`.

```bash
# Detector
python3 scripts/benchmark_speed.py --mode detector \
    --weights runs/obb/runs/baseline_yolo11m_dota_ships_obb/weights/best.pt \
    --imgsz 1024 --batch-size 8 --device cuda:1 \
    --warmup 10 --iters 50 --label yolo11m_obb \
    --out reports/speed/yolo11m_obb.json

# Gate (one per backbone)
python3 scripts/benchmark_speed.py --mode gate \
    --gate-config configs/experiments/gate_resnet18.yaml \
    --weights runs/gate_resnet18/best.pt \
    --imgsz 256 --batch-size 32 --device cuda:1 \
    --warmup 20 --iters 100 --label gate_resnet18 \
    --out reports/speed/gate_resnet18.json
```

Each run writes a JSON with `gflops_per_image`, `ms_per_image_gpu`, and
optionally `ms_per_image_cpu` (`--include-cpu`). Plug those numbers into the
matching `--gate-flops-g`/`--gate-latency-ms`/`--detector-flops-g`/
`--detector-latency-ms` flags of `eval_cascade.py` for a re-run that produces
publication-grade compute numbers.

Buckets are emitted as `<out>.stratified_<stratum>.csv`.

## Cross-dataset robustness

```bash
scripts/run_all.sh hrsc        # zero-shot HRSC2016 evaluation
```

## Reports & figures

```bash
scripts/run_all.sh report
# Then open notebooks/01_master_pareto.ipynb for the canonical figure.
```

## Repo layout

```
src/
  datasets/      vendored tiling/converters/validation + extended dota.py + hrsc2016.py
  experiments/   vendored Ultralytics trainer/evaluator/metrics/speed
  utils/         vendored paths/seed/logging/hardware/reproducibility/ultralytics_env
  splits.py      geographic split builder + filelist materialization
  sparsity.py    tile-level sparsity statistics
  tile_features.py  cheap per-tile features (RGB stats / Sobel / entropy / GSD bucket)
  gate.py        gate models, dataset, training, scoring (Phase 2 spine)
  calibration.py 6 calibration methods + ECE/MCE
  cascade.py     runtime composition (gate -> conditional detector)
  cascade_eval.py end-to-end mAP-at-compute + stratified eval + oracle gate
  codesign.py    shared backbone / early-exit / distillation / relaxed gating
configs/
  datasets/      dota_ships.yaml, hrsc2016_ships.yaml (rewritten by prepare scripts)
  splits/        geographic_v1.yaml
  experiments/   3 baselines + 6 gate configs
scripts/
  prepare_data.py        tile + binary + meta + sparsity + split (one entry)
  prepare_hrsc.py        HRSC2016 -> cascade layout
  train_detector.py      YOLO-OBB training wrapper
  train_gate.py          gate training
  train_codesign.py      Pillar 3 variants
  train_distill.py       detector -> gate distillation
  score_tiles.py         emit per-tile gate scores JSONL
  extract_tile_features.py  per-tile features for context-adaptive
  eval_cascade.py        per-config Pareto rows; supports --all-calibrations and --stratum
  benchmark_speed.py     measured GFLOPs + per-image latency (detector or gate)
  run_pareto.py          aggregate + plot
  run_experiment.sh      ships pipeline orchestrator with sentinel-based crash recovery
  run_experiment_class.sh  per-class pipeline orchestrator (planes / small_vehicle / arbitrary CLASS_TAG)
  run_all.sh             one-stage / full orchestration
notebooks/
  01_master_pareto.ipynb  master compute-accuracy figure
data/  runs/  reports/  tests/
```

## Tests

```bash
pytest tests/ -q
```

21 tests covering tiling, polygon clipping, header parsing, calibration math,
threshold strategies, cascade composition, polygon mAP, oracle gate, geographic
split disjointness, sparsity statistics, and tile-feature extraction.

## Status

Pillars 1–4 are complete end-to-end on three classes (`ships`, `planes`,
`small_vehicle`). Per-class outputs:

- 6 trained gate backbones + 1 YOLO11n/s/m baseline trio
- Cascade Pareto: 6 backbones × **5 calibrations** (identity / temperature /
  Platt / isotonic / **context_adaptive @ recall=0.95**) + oracle, on val split
- Stratified Pareto inside size / GSD / imagesource / boundary buckets,
  with per-bucket `filter_rate` and `gate_recall_on_positive_tiles`
- Co-design: `shared`, `early_exit`, `relaxed` variants trained with the
  gate-warmup schedule
- Distillation on all three classes (clean negative result, PR-AUC ≈ 0.10–0.30
  vs 0.78–0.94 for binary-trained gates)
- Measured FLOPs/latency via `scripts/benchmark_speed.py` (per-backbone
  GFLOPs and ms/img on Quadro RTX 8000 → `reports/speed/*.json`); cascade
  evals consume these directly instead of the previously hardcoded constants

## TODO before paper submission

- [ ] **HRSC2016 zero-shot cross-dataset eval** (research plan §7.4) — load-bearing
      held-out evaluation given the in-DOTA test-set limitation below.
- [ ] **In-DOTA test-set evaluation is currently not possible.** Skipped.
      `data/processed/dota_*/images/test/` was populated from DOTA-v1.5's official
      test split, which has no public ground truth (held by the benchmark server).
      All 10,833 test labels are zero-byte placeholders, so mAP cannot be computed
      on test and Platt/isotonic calibration fits crash with `"only one class: 0"`.
      The geographic split in `configs/splits/geographic_v1.yaml` was generated
      with 489 held-out test stems from DOTA train, but those stems were never
      actually used to materialize tiles. To enable in-DOTA test, either:
      (a) re-run `prepare_data.py` so the geographic split partitions DOTA train
          into train/val/test and re-train the YOLO11m detector on the
          geographic-train subset only (~40 h scope), or
      (b) report val-only and rely on HRSC2016 as the held-out test (standard
          DOTA practice for many published papers).
- [ ] **YOLO11m re-train with `patience=50`.** The current YOLO11m baselines
      early-stopped at epochs 11–14 because `patience=10` triggered before the
      ImageNet-pretrained weights had a chance to fine-tune. Re-running with
      `patience=50` would give a stronger full-pass baseline (~10 h/class on
      free GPU). Cascade ranking is unaffected (savings are relative), but the
      headline mAP numbers would improve.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) §14 checklist and the implementation
plan at `~/.claude/plans/here-i-have-a-starry-sunbeam.md` for the full
pre-submission audit.
