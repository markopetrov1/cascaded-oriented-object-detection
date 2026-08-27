# Cascaded Object Detection for Remote Sensing Imagery

**A research project plan and roadmap**

> This document is the canonical reference for the project. It describes what we're building, why it matters, what the actual research contribution is, how the work is structured, and how to get started. Read it end-to-end before writing any code. It is also intended to be consumed by Claude Code as project context, so keep it up to date as the work evolves.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Motivation and Real-World Impact](#2-motivation-and-real-world-impact)
3. [The Proposed Architecture](#3-the-proposed-architecture)
4. [Reality Check: What Is and Isn't Novel](#4-reality-check-what-is-and-isnt-novel)
5. [Research Thesis](#5-research-thesis)
6. [Background and Required Reading](#6-background-and-required-reading)
7. [The Four Research Pillars](#7-the-four-research-pillars)
8. [Datasets](#8-datasets)
9. [Experimental Methodology](#9-experimental-methodology)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Remote Sensing Specifics (Cross-Cutting Concerns)](#11-remote-sensing-specifics-cross-cutting-concerns)
12. [Project Setup](#12-project-setup)
13. [Repository Structure](#13-repository-structure)
14. [Roadmap and Milestones](#14-roadmap-and-milestones)
15. [Pitfalls and Common Mistakes](#15-pitfalls-and-common-mistakes)
16. [Deliverables](#16-deliverables)
17. [Resources](#17-resources)

---

## 1. Project Summary

We are studying **cascaded object detection** in **aerial / satellite (remote sensing) imagery**.

The pipeline has two stages:

1. **Stage 1 — The Gate.** A lightweight binary classifier that decides whether a tile contains any object of interest (e.g., a ship). If not, the tile is discarded immediately.
2. **Stage 2 — The Detector.** A full object detector (with oriented bounding boxes) that runs only on tiles the gate accepted.

The motivation is straightforward: in remote sensing imagery, the vast majority of image tiles contain no objects of interest. Running a heavy detector on every tile wastes compute. A cheap gate can reject most empty tiles, allowing the detector to spend its budget where objects actually exist.

This document describes a research project that **systematically characterizes when and how this approach pays off**, rather than simply building it.

---

## 2. Motivation and Real-World Impact

### 2.1 Why Remote Sensing Is Special

Remote sensing imagery has properties that make cascaded detection genuinely valuable rather than merely a curiosity:

- **Massive image dimensions.** A single satellite image can be 20,000 × 20,000 pixels or larger. Processing it requires tiling (e.g., into 1024 × 1024 patches), producing hundreds to thousands of tiles per image.
- **Extreme spatial sparsity.** For many object classes (ships in open ocean, aircraft on tarmac, oil tanks in industrial zones), most tiles contain *zero* objects. Empty-tile rates of 70–95% are common.
- **Long-tail class imbalance.** Most pixels are background. Most tiles are background. Most batches during training are background-dominated.
- **Multiple sensors and resolutions (GSD).** A single dataset like DOTA mixes imagery at different ground sampling distances. Methods must be robust across GSDs.
- **Oriented objects.** Objects in aerial imagery appear at arbitrary rotations, requiring oriented bounding boxes (OBB) rather than axis-aligned ones.
- **Compute constraints in deployment.** Many real applications (drones, edge processing on satellites, real-time monitoring) have hard latency budgets. Throughput matters, not just accuracy.

### 2.2 Real-World Applications

A working cascade detector for RS imagery has direct utility in:

- **Maritime monitoring.** Ship detection over open ocean for fisheries enforcement, search and rescue, smuggling interdiction. Ocean tiles are overwhelmingly empty — exactly the regime where cascading wins big.
- **Disaster response.** Rapid post-disaster imagery analysis (damaged buildings, blocked roads, vehicles) where compute time directly affects human outcomes.
- **Environmental monitoring.** Tracking deforestation, illegal mining, infrastructure construction over very large areas.
- **Defense and intelligence.** Wide-area surveillance with limited compute.
- **Agriculture and infrastructure inventory.** Counting silos, vehicles, or specific equipment across large regions.

In all of these, **wall-clock time per square kilometer** is the metric that matters operationally, and that's the metric a good cascade improves.

### 2.3 Why This Specific Project Matters

Cascade-style detection has been explored before in vision generally and even occasionally in RS. What is missing — and what makes this project worth doing — is a **systematic study** that answers:

- *When* does cascading help and *when* does it hurt? (As a function of object sparsity, image scale, gate quality.)
- *How* should the gate be designed and trained for maximum end-to-end benefit?
- *How* should the gate's decision threshold be calibrated against the *downstream detection metric*, not classification accuracy?
- *How* does this all interact with the realities of remote sensing (oriented boxes, tiny objects, GSD variance, geographic distribution shift)?

Producing principled, experimentally-grounded answers to these questions is the contribution.

---

## 3. The Proposed Architecture

### 3.1 The Pipeline

```
Large RS image (e.g., 8000×8000)
        │
        ▼
   [Tiling: 1024×1024 with overlap]
        │
        ▼
   ┌─────────────────────────┐
   │ Tile (1024×1024)        │
   └─────────────────────────┘
        │
        ▼
   ┌─────────────────────────┐
   │  Stage 1: Gate          │   ← lightweight classifier
   │  (binary: object?)      │     (e.g., ResNet-18, MobileNet,
   └─────────────────────────┘      EfficientNet-B0, small ViT)
        │
   ┌────┴────┐
   │         │
 negative   positive
   │         │
   ▼         ▼
 discard   ┌─────────────────────────┐
           │  Stage 2: Detector       │  ← OBB detector
           │  (oriented boxes +       │     (YOLO26-OBB,
           │   class predictions)     │      Oriented R-CNN, etc.)
           └─────────────────────────┘
                    │
                    ▼
            Detected objects
```

### 3.2 What "Cheap" and "Heavy" Mean Here

| Property             | Gate (Stage 1)                         | Detector (Stage 2)                          |
| -------------------- | -------------------------------------- | ------------------------------------------- |
| Task                 | Binary classification                  | Oriented object detection                   |
| Output               | One scalar (probability)               | Set of oriented boxes + classes + scores    |
| Typical params       | 1M – 25M                               | 25M – 250M                                  |
| Typical FLOPs        | Low                                    | High                                        |
| Failure mode         | False negative → missed detection      | False positive / poor localization          |
| Critical metric      | Recall on positive class               | mAP@0.5, mAP@0.5:0.95                       |

The crucial asymmetry: **a false negative at the gate is an unrecoverable miss for the entire pipeline.** This shapes everything about how the gate is designed, trained, and thresholded.

---

## 4. Reality Check: What Is and Isn't Novel

Before claiming novelty, students must understand that cascaded detection is an old idea that exists under several names:

- **Viola–Jones (2001):** the original cascade — a series of progressively more expensive classifiers.
- **Cascade R-CNN:** progressive box refinement (different concept but related vocabulary).
- **Early-exit networks / BranchyNet:** networks that exit early when confident.
- **Region Proposal Networks:** internally a cheap "is anything here?" stage.
- **Empty-tile rejection / patch screening in remote sensing:** has appeared in scattered papers.

**The architecture itself is not the contribution.** If the project is pitched as "we built a cascade," it will be rejected.

The contributions live in the *systematic study* of:

- The Pareto frontier between accuracy and compute, as a function of sparsity.
- How to co-design gate and detector (rather than train independently).
- How to calibrate the gate's threshold against downstream detection metrics.
- How all of this interacts with remote sensing specifics.

Students should internalize this framing and lead with the *study*, not the architecture.

---

## 5. Research Thesis

> **In remote sensing imagery — where extreme spatial sparsity, multi-resolution sensors, oriented small objects, and severe class imbalance are the norm — the compute–accuracy frontier of cascaded object detection is governed by three controllable factors: (1) the operating sparsity regime, (2) classifier–detector co-design, and (3) gate threshold calibration against downstream detection metrics. We characterize each factor systematically and produce practical guidance for when and how to deploy cascaded detection in RS pipelines.**

This thesis statement should be the spine of every result, every plot, every paragraph.

---

## 6. Background and Required Reading

Students must read these *before* writing significant code. Group them by week.

### 6.1 Foundations of Object Detection (Week 1)

- **Faster R-CNN** (Ren et al., 2015) — anchor boxes, RPN.
- **YOLO series overview** — focus on YOLO11 and YOLO26 (the OBB head matters).
- **Focal Loss / RetinaNet** (Lin et al., 2017) — class imbalance.
- **DETR** (Carion et al., 2020) — set-based detection, attention.

### 6.2 Cascade and Early-Exit Methods (Week 1–2)

- **Viola–Jones** (2001) — the original cascade.
- **Cascade R-CNN** (Cai & Vasconcelos, 2018).
- **BranchyNet** (Teerapittayanon et al., 2016) — early-exit deep networks.
- **Survey papers on dynamic neural networks / conditional computation.**

### 6.3 Remote Sensing Detection (Week 2)

- **DOTA dataset paper** (Xia et al., 2018) and DOTA v2.0 update.
- **Oriented R-CNN** (Xie et al., 2021).
- **ReDet** (Han et al., 2021) — rotation-equivariant detection.
- **RoI Transformer** (Ding et al., 2019).
- **Oriented R-CNN, S2A-Net, R3Det** — the standard OBB family.
- **HRSC2016** (Liu et al., 2017) — ship-specific benchmark.
- **DIOR** (Li et al., 2020) — large-scale RS detection.
- **FAIR1M** — fine-grained RS detection.

### 6.4 Tile-Level Filtering / Patch Screening (Week 2)

Search terms for the literature dive:

- "empty tile rejection"
- "patch classification before detection"
- "coarse-to-fine aerial detection"
- "saliency-guided detection remote sensing"
- "screening network object detection"

There is no canonical paper here — it's a scattered literature, and surveying it well is itself a contribution.

### 6.5 Calibration and Thresholding (Week 3)

- **Temperature scaling** (Guo et al., 2017).
- **Platt scaling, isotonic regression** — classical calibration.
- **Calibration of modern neural networks** — recent literature on calibration under distribution shift.

### 6.6 RS Foundation Models (Week 3, optional but recommended)

- **SatMAE, Prithvi, SkySense, Scale-MAE** — pretrained backbones for RS.
- Useful for the co-design pillar (Pillar 3) — comparing ImageNet-pretrained vs. RS-pretrained backbones.

---

## 7. The Four Research Pillars

The project is structured around **three experimental pillars** (1, 2, 3), with the **fourth (RS specifics)** as a cross-cutting lens applied to all of them.

### 7.1 Pillar 1 — Pareto / Economics

**Question:** When does cascading actually pay off?

**Headline figure:** mAP vs. compute (FLOPs or latency) for the baseline detector and many cascade variants, swept across object sparsity regimes.

**Independent variables:**
- Object class (which controls tile-level positive rate, i.e., sparsity).
- Gate model capacity (ResNet-18, ResNet-50, MobileNetV3-small, MobileNetV3-large, EfficientNet-B0, a tiny custom net ~1M params, optionally a small ViT or RS-foundation-model linear probe).
- Gate threshold.
- Detector choice (YOLO26-OBB and one heavier alternative, e.g., Oriented R-CNN).

**Dependent variables:**
- End-to-end mAP (overall and per-class).
- End-to-end FLOPs per image.
- End-to-end latency (median, p95, p99) on a fixed hardware target.
- Filter rate (fraction of tiles rejected).

**Critical baselines (do not skip):**
- **Oracle gate.** A perfect classifier (uses ground truth). Sets the upper bound on cascade quality at zero gate compute. Without this number, no one can judge whether the real gate is good.
- **Matched-compute single-stage detector.** A smaller standalone detector tuned to use roughly the same total compute as the cascade. If this matches or beats the cascade, the cascade isn't earning its complexity.

**Headline contribution:** a *rule of thumb* for when cascading helps in RS, expressed in terms of measurable dataset properties (tile-level positive rate, average objects per positive tile, etc.).

### 7.2 Pillar 2 — Calibration

**Question:** How should the gate's decision threshold be set to optimize end-to-end detection metrics, not classification accuracy?

**Methods to compare (in order of sophistication):**
1. **Naïve.** Threshold = 0.5.
2. **Recall-targeted.** Threshold chosen on validation to hit a fixed recall floor (e.g., 99% on positive tiles).
3. **End-to-end mAP-optimal.** Grid search threshold to maximize end-to-end mAP on validation.
4. **Score-calibrated then thresholded.** Apply temperature scaling / Platt / isotonic regression to gate outputs first, then threshold.
5. **Context-adaptive thresholds.** Threshold varies as a function of cheap tile features (mean color, texture statistics, scene type, geographic priors). RS-specific and novel.
6. **Learned thresholds.** A tiny MLP mapping (gate logits, cheap tile features) → threshold, trained against a differentiable surrogate of mAP.

**Headline figure:** end-to-end mAP at fixed compute budget, across calibration methods.

**Headline contribution:** an empirical and principled treatment of gate threshold calibration in RS pipelines, including the context-adaptive variant.

**Subtle point to internalize:** the gate's metric is *not* accuracy or F1. It is downstream-mAP-at-given-compute. Train students to think this way from day one.

### 7.3 Pillar 3 — Co-design

**Question:** Can joint design of gate and detector move the Pareto frontier?

**Configurations to compare against the independently-trained baseline:**
1. **Shared backbone, two heads.** One backbone shared between gate (binary head) and detector (detection head). Joint training with weighted loss. Tests whether the gate gets cheaper without losing quality when piggybacking on detector features.
2. **Early-exit detector.** The gate is literally the first N blocks of the detector backbone, with a binary head. If positive, computation continues. The most compute-efficient variant because shared layers aren't recomputed at Stage 2. Implementation is fiddlier (requires caching activations).
3. **Distillation: detector → gate.** Train an independent gate using the detector's max-objectness-per-tile as a soft target. Hypothesis: detector confidence carries richer signal than hard binary labels.
4. **End-to-end with relaxed gating.** Differentiable gate (Gumbel-softmax or straight-through estimator) so gradient flows from detection loss back through the gate. Risk: training instability. Worth attempting; if it works, it's the strongest single result in the paper.

**Bonus sub-experiment:** ImageNet-pretrained vs. RS-foundation-model-pretrained backbones, all else equal. Likely a meaningful effect in RS.

**Headline figure:** Pareto frontiers of all co-design variants overlaid against the independently-trained baseline.

**Headline contribution:** demonstration that co-design shifts the *frontier*, not just a single operating point — with mechanistic explanation of *why*.

### 7.4 Pillar 4 (Lens) — Aerial / OBB Specifics

Not a separate pillar, but a set of analyses to apply within each of Pillars 1–3.

- **Tile boundary objects.** Report metrics separately for objects fully inside vs. crossing tile boundaries.
- **GSD stratification.** Bucket tiles by ground sampling distance and report per-bucket performance.
- **Geographic generalization.** Train/val/test splits by *region*, not random tiles.
- **Object-scale stratification.** Separate AP for very small (<32×32 px), small, medium, large objects.
- **OBB consistency.** Identical OBB detection heads across cascade and baselines.
- **Class imbalance handling in gate.** Compare BCE, focal loss, weighted sampling, ratio-controlled batches — and observe how each interacts with calibration (Pillar 2).

These analyses appear in the paper as supplementary tables and stratified Pareto plots.

---

## 8. Datasets

### 8.1 Primary: DOTA v2.0

- **Source:** [https://captain-whu.github.io/DOTA/](https://captain-whu.github.io/DOTA/)
- **Size:** ~11,268 images, ~1.8M instances across 18 classes (DOTA v2.0).
- **Annotation type:** Oriented bounding boxes.
- **Why:** the canonical large-scale OBB benchmark. Class sparsity varies enormously, which is what we need for the Pareto study.

**Class scope decision (commit to this early):**
- **Single positive class (recommended start):** ships only. Cleanest binary gate problem, well-motivated by maritime applications.
- **Multi-class generalization (later):** treat positive-class-of-interest vs. all-others, or train one model per sparse class and aggregate.

### 8.2 Secondary: HRSC2016

- **Source:** ship-detection-only benchmark.
- **Why:** validates ship-specific results from DOTA on a different distribution. Strengthens any ship-focused claim.

### 8.3 Tertiary: DIOR or FAIR1M

- For cross-dataset robustness checks. Use sparingly — not for primary results.

### 8.4 Why Not COCO

COCO has roughly zero empty images for any common class. The cascade question is uninteresting on COCO. Mentioning this explicitly in the paper helps frame *why* this is an RS problem, not a general detection problem.

---

## 9. Experimental Methodology

### 9.1 Stage 0: Data Preparation

This will consume ~30% of the project time. Plan accordingly.

1. **Download DOTA v2.0** training, validation, and test-dev splits.
2. **Tile each image.** Standard: 1024×1024 with 200px overlap. Tile size is itself a hyperparameter — ablate at least one alternative (e.g., 800×800).
3. **Generate two label sets per tile:**
   - **Binary label** for the gate: `1` if tile contains ≥ 1 instance of any positive class, else `0`.
   - **OBB labels** for the detector: standard DOTA format.
4. **Compute tile statistics.** Per class, report:
   - Total tiles.
   - Tiles with ≥ 1 instance (positive rate).
   - Mean and median objects per positive tile.
   - Object size distribution.
   - These statistics are themselves a contribution and motivate the rest of the study.
5. **Create splits:**
   - **Random tile split** (standard, weak).
   - **Geographic / image-level split** (strong, RS-correct). Train on a set of source images; val/test on disjoint source images, preferably from different geographic regions.
   - Run all primary experiments on the geographic split. The random split is a reference only.

### 9.2 Stage 1: Baselines

Train and evaluate, with no cascade:

- **YOLO26-OBB** at multiple sizes (n, s, m, l) on the tiled data.
- **Oriented R-CNN** as a stronger but slower comparison.

Report for each:

- mAP@0.5, mAP@0.5:0.95, per-class AP.
- Latency (ms/tile) on the target hardware (specify GPU model in the paper).
- Throughput (tiles/sec).
- GFLOPs per tile.

These are the bars to beat.

### 9.3 Stage 2: Gate Training

For each gate backbone (ResNet-18, MobileNetV3, EfficientNet-B0, …):

- Train as binary classifier on tile labels.
- Compare class-imbalance handling (BCE, focal loss, weighted sampling).
- Report:
  - Accuracy, precision, recall, F1.
  - Recall–FLOP tradeoff curve at varying thresholds.
  - Calibration metrics (ECE, MCE, reliability diagrams).

The **recall-FLOP curve** is the deliverable of this stage, not a single number.

### 9.4 Stage 3: Cascade Composition

- Pipeline: tile → gate → if positive, detector → OBB outputs.
- Apply each calibration method from Pillar 2.
- For every (gate × calibration × detector) combination, measure end-to-end metrics.
- Plot the Pareto frontier (mAP vs. compute) and overlay all configurations.

### 9.5 Stage 4: Co-design Experiments (Pillar 3)

Each co-design configuration (shared backbone, early-exit, distillation, end-to-end relaxed gating) gets its own training run and its own Pareto curve. Overlay against the independent-training baseline. Key plot: do the curves *shift*, or do they only move along an existing curve?

### 9.6 Stage 5: RS-specific Stratifications (Pillar 4 lens)

Re-run analyses, stratifying by:
- Tile-boundary vs. interior objects.
- GSD bucket.
- Object size bucket.
- Geographic region.

Report stratified Pareto plots in the supplement.

### 9.7 Stage 6: Cross-Dataset Robustness

Apply the best cascade configuration trained on DOTA-Ships to HRSC2016 (zero-shot). Measure degradation. This is a robustness check, not a primary result.

---

## 10. Evaluation Metrics

Report all of these. Always.

| Metric | Why |
| ----------------------------------- | ---------------------------------------------- |
| mAP@0.5                             | Standard detection accuracy.                   |
| mAP@0.5:0.95                        | Stricter localization quality.                 |
| Per-class AP                        | Hides class-specific failures otherwise.       |
| Filter rate                         | Fraction of tiles rejected by gate.            |
| Gate recall on positive class       | Hard ceiling on cascade detection recall.      |
| Gate ECE / MCE                      | Calibration quality.                           |
| Latency per image (median)          | Operational speed.                             |
| Latency p95, p99                    | Tail behavior — matters for real-time systems. |
| GFLOPs per image (averaged)         | Compute use including gate cost.               |
| Stratified AP (size, GSD, region)   | RS-specific honesty.                           |

The **mAP-vs-compute Pareto plot** is the central figure of the paper. Without it, results don't mean much.

---

## 11. Remote Sensing Specifics (Cross-Cutting Concerns)

These apply to *every* experiment.

- **Geographic train/test splits, not random.** A random tile split lets neighboring tiles from the same source image leak between train and test. Always split by source image, ideally by geographic region.
- **GSD stratification.** Different sensors have different ground sampling distances. Cascade behavior varies meaningfully across GSD — report it.
- **Tile boundaries.** Objects crossing tile edges are uniquely hard. Track boundary-object metrics separately.
- **Tiny objects.** A non-trivial fraction of DOTA objects are below 32×32 pixels. Standard COCO-style "small" thresholds are inadequate; use finer-grained size buckets.
- **OBB-specific traps.** Angular periodicity around 0/180°, very thin elongated objects (ships especially). Use established OBB detection heads (YOLO26-OBB, Oriented R-CNN); do not roll your own.
- **Imbalance during gate training.** Negatives outnumber positives extremely. Use a controlled positive-to-negative ratio in batches (e.g., 1:3 or 1:4) rather than naïve uniform sampling.
- **Foundation model considerations.** ImageNet-pretrained backbones are mismatched to aerial imagery. Where feasible, compare against an RS-pretrained backbone (SatMAE, Prithvi, SkySense, Scale-MAE).

---

## 12. Project Setup

### 12.1 System Requirements

- **OS:** Linux (Ubuntu 22.04+) recommended. WSL2 acceptable.
- **GPU:** at least one CUDA GPU with ≥ 16GB VRAM for detector training. 24GB+ preferred. Multi-GPU helpful for the co-design pillar.
- **Disk:** ≥ 500GB for DOTA + tiles + checkpoints.
- **Python:** 3.10+.

### 12.2 Environment Setup

```bash
# create environment
conda create -n cascade-rs python=3.10 -y
conda activate cascade-rs

# core deep learning stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# detection libraries
pip install ultralytics              # YOLO26 with OBB support
pip install mmcv-full mmdet mmrotate # for Oriented R-CNN, ReDet etc. (optional but recommended)

# data, geo, RS
pip install opencv-python pillow numpy pandas
pip install shapely rasterio         # geo / OBB geometry
pip install pyproj                   # coordinate transforms

# experiment tracking and utilities
pip install wandb tensorboard
pip install hydra-core omegaconf     # config management
pip install scikit-learn matplotlib seaborn
pip install pycocotools

# calibration utilities
pip install netcal                   # calibration metrics and methods

# dev
pip install pytest black ruff mypy
```

### 12.3 Dataset Setup

```bash
# register at the DOTA website, then download
# https://captain-whu.github.io/DOTA/dataset.html

mkdir -p data/dota_v2/{train,val,test}/{images,labels}
# place downloaded files appropriately

# one script handles tiling, binary label generation, and sparsity stats
python scripts/prepare_data.py \
    --input data/dota_v2 \
    --output data/dota_v2_tiled \
    --tile-size 1024 \
    --overlap 200 \
    --positive-classes ship \
    --stats-out reports/sparsity_stats.json
```

### 12.4 Verifying the Setup

```bash
# train a quick YOLO26-OBB baseline (small, few epochs) to confirm everything works
python scripts/train.py --config configs/baseline_yolo26n.yaml --epochs 5
```

If this completes and produces reasonable mAP, the pipeline is ready.

---

## 13. Repository Structure

Keep it flat. Don't split things across many files until they actually need it.

```
cascaded-rs-detection/
├── README.md                # quick start, points to PROJECT_PLAN.md
├── PROJECT_PLAN.md          # this document
├── pyproject.toml
├── configs/                 # one YAML per experiment, flat
│   ├── baseline_yolo26n.yaml
│   ├── baseline_yolo26m.yaml
│   ├── gate_resnet18.yaml
│   ├── gate_mobilenetv3.yaml
│   ├── cascade_independent.yaml
│   ├── cascade_shared_backbone.yaml
│   ├── cascade_early_exit.yaml
│   └── cascade_distillation.yaml
├── src/
│   ├── data.py              # tiling, datasets, splits, sparsity stats
│   ├── models.py            # gates, detector wrappers, cascade variants
│   ├── calibration.py       # all calibration methods in one file
│   ├── train.py             # training entry points (gate, detector, cascade)
│   ├── eval.py              # metrics, Pareto, stratified analysis, oracle
│   └── utils.py             # OBB geometry, FLOPs, seeding
├── scripts/
│   ├── prepare_data.py      # tiling + binary labels + sparsity stats
│   ├── train.py             # dispatches by config
│   └── run_study.py         # Pareto / calibration / co-design sweeps
├── notebooks/
│   └── analysis.ipynb       # one notebook, sectioned; split only if it gets unwieldy
├── reports/                 # generated outputs: stats, Pareto data, figures
└── runs/                    # checkpoints and logs (gitignored)
```

Notes on the shape:

- **One file per concept, not per class.** `models.py` holds gates, detector wrappers, and cascade variants together. They're a few hundred lines each and they evolve together — splitting them across folders just creates import noise.
- **Flat configs.** One YAML per experiment beats nested config trees for a project this size. When you have 50+ configs, revisit this.
- **`scripts/` stays small.** Three entry points cover everything. If a fourth is needed, add it; don't preemptively create one per pillar.
- **Split when files actually hurt.** If `models.py` crosses ~1000 lines or `calibration.py` grows beyond what one person can hold in their head, split *then*. Premature splitting is the more common failure mode in research code.

---

## 14. Roadmap and Milestones

A 14-week timeline for a 3-student team. Adjust as needed.

### Phase 1 — Foundations (Weeks 1–3)

**Everyone, in parallel.**

- **Week 1.** Read foundations + cascade literature. Set up environments. Get DOTA downloaded.
- **Week 2.** Read RS detection + tile-filtering literature. Implement and run tiling pipeline.
- **Week 3.** Read calibration literature. Generate binary labels, compute and document sparsity statistics. Train YOLO26-OBB baseline; reproduce reasonable mAP.

**Phase 1 deliverable:** clean tiled dataset, sparsity report, working baseline detector, splits (random and geographic), shared codebase scaffold.

### Phase 2 — First-Cut Pillar Results (Weeks 4–6)

Each student takes one pillar. Shared infrastructure (gate training, evaluation harness) is built collaboratively.

- **Student A (Pillar 1, Pareto).** Train a few gate sizes. Produce first Pareto plot for ships on geographic split.
- **Student B (Pillar 2, Calibration).** Implement calibration methods 1–4. Compare on the strongest gate from Student A's set.
- **Student C (Pillar 3, Co-design).** Implement shared-backbone variant. First co-design Pareto curve.

**Phase 2 deliverable:** three rough Pareto plots, one per pillar. Code paths for each pillar working end-to-end.

### Phase 3 — Depth (Weeks 7–10)

- **Student A.** Add more gate variants, the oracle baseline, the matched-compute single-stage baseline. Begin Pillar 4 stratifications (size, GSD, geography).
- **Student B.** Implement context-adaptive thresholds (method 5) and learned thresholds (method 6). Calibration analyses across all gate variants.
- **Student C.** Implement early-exit detector and distillation variants. Begin end-to-end relaxed-gating experiment.

Calibration student typically finishes first (cheapest pillar) and then helps co-design student (most expensive).

**Phase 3 deliverable:** mature Pareto, calibration, and co-design results with proper baselines.

### Phase 4 — Integration (Weeks 11–13)

- Combined Pareto plot showing best operating points across all three pillars.
- Apply Pillar 4 stratifications across all experiments.
- Cross-dataset robustness check on HRSC2016.
- Begin paper drafting.

**Phase 4 deliverable:** one integrated narrative, one set of canonical figures.

### Phase 5 — Writing and Polish (Weeks 14+)

- Ablations to defend specific design choices.
- Negative-result tables (things that didn't work — these strengthen a paper).
- Final figures and tables.
- Paper draft.

### Quick Checklist of "Have We Done This?" Before Submitting

- [ ] Oracle gate baseline is reported.
- [ ] Matched-compute single-stage baseline is reported.
- [ ] Geographic splits are used for primary results.
- [ ] Pareto plots include all gate-detector combinations.
- [ ] Calibration methods include at least one context-adaptive variant.
- [ ] Co-design includes at least early-exit *or* shared-backbone with ablations.
- [ ] Stratified results (size, GSD, boundary) appear in supplement.
- [ ] Cross-dataset (HRSC2016) result appears.
- [ ] Latency reported on a specified hardware target.
- [ ] FLOPs reported and consistent with latency story.
- [ ] Negative results documented.

---

## 15. Pitfalls and Common Mistakes

### 15.1 The Recall Ceiling

Every false negative at the gate is an unrecoverable miss. If the gate has 95% recall on positive tiles, end-to-end recall is capped at 95%. Students will repeatedly underweight this. The fix: track gate recall as a primary metric on every plot, not a footnote.

### 15.2 Random Splits

Tiles from the same source image are highly correlated. A random tile split gives optimistic numbers that don't reflect deployment. Always use geographic / source-image-disjoint splits for primary results.

### 15.3 Sparsity-Dependent Conclusions

Cascading looks great on DOTA-Ships (mostly empty tiles). It would look terrible on a dense urban-vehicle dataset. Be explicit about this. The paper's contribution is precisely the *characterization* of this dependence — hiding it weakens the paper.

### 15.4 Apples-to-Apples Comparison

The cascade must be benchmarked against the *same detector* run end-to-end on the *same hardware* with the *same input pipeline*. Latency comparisons across different machines are meaningless. Pin one hardware target and run everything there.

### 15.5 Forgetting the Matched-Compute Baseline

If a smaller standalone detector can match cascade performance at the same FLOPs, the cascade adds complexity for nothing. Reviewers will ask for this comparison. Provide it up front.

### 15.6 OBB Implementation Bugs Confounding Results

Angular periodicity and very thin elongated boxes are tricky. Use established OBB heads. Confirm the baseline's OBB performance matches published numbers before introducing the cascade. Otherwise, observed differences may be OBB bugs, not cascade effects.

### 15.7 Treating the Gate as a Standard Classifier

The gate's job is *not* to be accurate. It is to maximize end-to-end mAP at given compute. Optimize and select gates by that metric, not by accuracy or F1.

### 15.8 Calibration Forgotten

Modern deep classifiers under heavy class imbalance are typically badly calibrated. Without explicit calibration, threshold-tuning becomes a confounding mess. Report ECE/MCE for every gate.

### 15.9 Optimistic Latency Numbers

Latency includes data loading, tiling, gate inference, conditional detector inference, and post-processing. Median is not enough; report p95 and p99 because tail latency is what kills real-time systems.

### 15.10 Skipping Negative Results

Things that didn't work (e.g., end-to-end relaxed gating that wouldn't converge) make the paper stronger, not weaker. Document failures clearly. They guide future work and demonstrate experimental discipline.

---

## 16. Deliverables

### 16.1 Code

- A single, well-documented repository implementing all three pillars.
- A README that points new contributors to this `PROJECT_PLAN.md`.
- Reproducible scripts: one command per primary figure.
- Configs (Hydra) for every reported experiment.
- Pretrained checkpoints for the headline results.

### 16.2 Reports / Figures

- **`reports/sparsity_stats.json` and a sparsity table** in the paper.
- **Master Pareto plot** combining all pillars.
- **Per-pillar Pareto plots.**
- **Calibration reliability diagrams** for each gate.
- **Stratified result tables** in the supplement.

### 16.3 Paper

- Target venue: a remote sensing journal (TGRS, ISPRS J. Photogrammetry, RSE) or a vision conference if results are strong (CVPR / WACV workshops on EarthVision, CVPRW on geospatial AI). Workshops are realistic targets for student-driven work.
- Strict adherence to the thesis statement in §5.
- Lead with the *systematic study*, never the architecture.

### 16.4 Internal

- A short tech-report-style summary (~5 pages) for archival within the lab regardless of external publication outcome.
- Clear handoff documentation for any future student picking up the work.

---

## 17. Resources

### 17.1 Datasets

- DOTA: https://captain-whu.github.io/DOTA/
- HRSC2016: search "HRSC2016 ship detection"
- DIOR: search "DIOR remote sensing detection benchmark"
- FAIR1M: search "FAIR1M fine-grained remote sensing"

### 17.2 Code and Tools

- Ultralytics (YOLO26, OBB support): https://github.com/ultralytics/ultralytics
- MMRotate (OBB detection toolbox): https://github.com/open-mmlab/mmrotate
- MMDetection (general detection toolbox): https://github.com/open-mmlab/mmdetection
- DOTA devkit (evaluation, OBB utilities): https://github.com/CAPTAIN-WHU/DOTA_devkit
- netcal (calibration metrics): https://github.com/EFS-OpenSource/calibration-framework

### 17.3 Compute

Specify your lab's available hardware here, and pin one GPU model as the canonical latency-reporting target so all numbers in the paper are comparable.

### 17.4 Lab Conventions

- All experiment runs go through Hydra configs and W&B logging.
- All commits reference an issue or experiment ID.
- All figures are reproducible from a single notebook in `notebooks/`.
- All checkpoints used in the paper are stored with their config.

---

## Final Note for Students

The single most common failure mode of student projects in this area is to build the cascade, run a few experiments, and write up "we built a cascade and it works." That is not a contribution. The contribution is the **systematic characterization of when, how, and why cascading helps in remote sensing detection**. Every figure, every paragraph, every experiment should be in service of that. If you find yourself producing results that don't fit the thesis in §5, the thesis is the thing to interrogate first — but more often, the experimental design is.

Read this document end-to-end before starting. Re-read it at the end of each phase. Update it when reality forces a change in plan.

Good luck.
