# Cascaded oriented object detection

Code and results for *Rethinking Reported Speedups in Cascaded Oriented Object
Detection* (Petrov, Pandilova, Trajanovski, Dimitrovski, Kitanovski).

The paper argues that the speedup a tile gate can achieve is set by the data
rather than by the gate. A cascade cannot skip more of the detector's work than
the share of tiles that are empty, and that share is measurable before any gate
is trained. This repository contains the pipeline that produced every number in
the paper, and the scripts that turn those results back into the manuscript's
tables and figures.

## What is here

```
src/                 tiling, gate models, calibration, cascade evaluation, OAN head
scripts/             one entry point per stage, plus the paper generators
configs/             dataset and experiment configs (per-class and generated)
reports/             all derived results: cascade sweeps, speed, calibration, figures
docs/                a note for co-authors on how the argument changed
```

The manuscript sources are not in this repository. What is here is the code and
the derived results the manuscript is built from, which is what a reader needs in
order to check it. The generators below write their output into a local `paper/`
directory, creating it if necessary. Training logs are not tracked either; they
are large and machine specific.

## Reproducing the paper without a GPU

Everything the manuscript claims is derived from `reports/`, which is committed.
If you only want to check that the paper's numbers follow from the released
results, no GPU and no dataset are needed:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/savings_model.py          # the cost identity, checked against every run
python scripts/sparsity_ceiling.py       # the envelope, 19 gating tasks
python scripts/prior_work_prediction.py  # published speedups read through the envelope
python scripts/flops_latency_gap.py      # arithmetic against wall-clock
python scripts/paper_numbers.py          # -> paper/generated_numbers.tex
python scripts/paper_tables.py           # -> paper/generated_tables.tex
python scripts/paper_tables_detail.py    # -> paper/generated_tables_detail.tex
PYTHONPATH=scripts python scripts/paper_figures_law.py
```

Running that chain twice reproduces every generated artifact byte for byte, which
is the property the manuscript relies on when it claims its numbers are not typed
by hand.

`savings_model.py` is the one to run first. It recovers every term of the cost
identity for all measured operating points and checks the result against the
recorded compute. It prints the worst residual, which should be at
floating-point level. If it is not, something upstream is wrong and nothing
derived from it should be trusted.

## Reproducing the experiments from scratch

This needs the imagery and a GPU. We used a single NVIDIA Quadro RTX 8000.

Paths in `configs/datasets/*.yaml` and in the metadata fields of
`reports/speed/*.json` carry a `<REPO_ROOT>` placeholder rather than the machine
they were produced on. `prepare_data.py` rewrites the dataset configs with real
paths when you tile the imagery, and nothing numeric depends on the placeholder.

**Data.** DOTA-1.5 and HRSC2016 are not redistributed here. Download them from
their maintainers and place DOTA at `data/raw/DOTA/{train,val,test}/` with
`images/` and `labelTxt/` subdirectories, and HRSC2016 at `data/raw/HRSC2016/`.
The tiling script reproduces our exact tile set from them at the tile size and
overlap below.

**Checkpoints.** Detector and gate weights are not tracked here, for size. They
are available from the corresponding author on request.

**Tiling.** Tiles are 1024 pixels square with 200 pixels of overlap, which is the
standard DOTA protocol and the same one the systems we compare against use. Tile
size and overlap change the positive-tile rate directly, so changing them changes
the envelope; keep them fixed when comparing against our numbers.

```bash
python scripts/prepare_data.py --raw-dir data/raw/DOTA \
    --out-dir data/processed/dota_ships --tile-size 1024 --overlap 200 \
    --positive-classes ship \
    --dataset-yaml configs/datasets/dota_ships.yaml \
    --stats-out reports/sparsity_ships.json
```

The tile grid does not depend on which class you gate for, so one tiled copy
serves every gating task. `scripts/make_gate_labels.py` derives the per-class
binary labels from the tiling metadata in seconds rather than re-tiling, and
`scripts/make_gate_configs.py` writes one training config per task.

**Detector.** One multi-class YOLO11m-OBB serves all gating tasks:

```bash
python scripts/train_detector.py \
    --config configs/experiments/baseline_yolo11m_multiclass_obb.yaml --device 0
```

Note that the public `yolo11m-obb` weights are already DOTA-pretrained, so this
fine-tunes rather than trains from scratch. Use `patience=50`; at the default of
10 the run stops around epoch 12 while the pretrained features are still
adapting.

**Gates and sweep.**

```bash
python scripts/make_gate_labels.py --all --min-positive-val 100
python scripts/make_gate_configs.py
scripts/run_unattended_sweep.sh          # detector pass, then one eval per task
```

The cascade evaluation caches detector predictions and polygon IoU once per
tile, so a threshold sweep costs no further inference. That is what makes a
19-task sweep affordable.

**Fused gate.** The reimplementation of the objectness head of Xie et al.
(*Sci. China Inf. Sci.* 2023) lives in `src/oan.py`:

```bash
python scripts/train_oan.py --data configs/datasets/dota_ships.yaml \
    --weights yolo11m-obb.pt --epochs 50 --patience 50 --lambda-oan 3.0 --device 0
python scripts/score_tiles_oan.py --weights <run>/weights/best.pt \
    --data-root data/processed/dota_ships --out reports/oan/gate_oan_ships_val.jsonl
```

Compare against a detector trained for the same number of epochs. The per-class
baselines elsewhere in this repository stopped early under `patience=10`, and
comparing a 50-epoch joint model against them confounds co-design with training
budget.

## Things that will trip you up

The Ultralytics user settings may point `runs_dir` somewhere unwritable. Every
entry point calls `configure_ultralytics_settings()` before importing
ultralytics, which redirects it to a project-local file; if you write a new
script, do the same, and do it *before* the import.

In Ultralytics training, `save` controls model checkpointing, not image saving.
Setting it false produces a run with an empty `weights/` directory. The large
preview JPEGs come from `predict save=True`, which is a different flag on a
different mode.

Ultralytics resolves `project` against its own `runs_dir`, so a run does not
necessarily land at `<project>/<name>`. `scripts/train_oan.py` records its
resolved output directory to `runs/oan_last_save_dir.txt` for this reason.

Ultralytics resolves image paths before deriving label paths, so a directory
symlink from one dataset's `images/` to another's will silently pull labels from
the symlink target. Use hard links, which share the inode and cannot be
redirected.

## Citation

A BibTeX entry will be added once the paper has a venue. Until then, cite the
repository.
