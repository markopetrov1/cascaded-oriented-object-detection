#!/usr/bin/env python3
"""Round-1 cascade evaluation: image-level, cross-fitted, with bootstrap intervals.

For one detector cache (scripts/r1_build_cache.py) and one gating task:

1. Calibration maps are fitted on fold A of DOTA val (configs/splits/r1_val_2fold.yaml).
2. Every (gate, calibration, threshold) candidate is scored on fold A. Thresholds
   are 200 quantiles of the calibrated fold-A scores rather than 21 fixed values.
3. The cheapest candidate whose image-level mAP@0.5 stays within --budget of
   fold A's full-pass mAP is selected (--metric: 101-point interpolation by
   default, as in the first submission's evaluator; the DOTA devkit's 11-point
   VOC07 AP is coarse above recall 0.9 and hides recall lost there), with cost given by
   --cost-model: decoupled (g + a) or fused (p + h + a(1 - p)).
4. That calibrator and threshold are applied to fold B, which is reported.
5. Folds swap; the two report-fold decisions together cover every tile once,
   and that pooled, cross-fitted operating point is the headline. Cluster
   bootstrap over images gives 95 % intervals.

The old protocol (select and report on the same tiles) is also computed on all
of val and labelled "same_split", for comparison only.

Presets define the tasks; see PRESETS. Output: reports/r1/cascade/<preset>/<task>.json
and <task>_masks.npz (accepted tiles of every reported operating point).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.calibration import build_calibration, fit_context_adaptive  # noqa: E402
from src.r1_eval import (CLASS_ID, IOU_THRESHOLDS, BootstrapMAP, EvalCache, evaluate, match,  # noqa: E402
                         merged_keep)

CALIBS = ["identity", "temperature", "platt", "isotonic", "context_adaptive"]
FOLD_FILE = REPO / "configs/splits/r1_val_2fold.yaml"
META = REPO / "data/processed/dota_multiclass/metadata/tiles.jsonl"
BB = ["mbv3small", "mbv3large", "resnet18", "resnet50", "effb0", "tiny"]
# Per-tile costs (thop, GFLOPs; gates at 256 px, detectors at 1024 px), all from the one
# measurement protocol in scripts/r1_benchmark.py when its output exists. The fallback
# values are the first submission's, kept so the script runs before the benchmark has.
GATE_GFLOPS = {"mbv3small": 0.1444, "mbv3large": 0.5609, "resnet18": 4.7635,
               "resnet50": 10.7930, "effb0": 1.0043, "tiny": 5.0041, "sweep_mbv3large": 0.6069}
DET_GFLOPS = {"multiclass": 184.17, "shipsp50": 184.02, "yolo11n": 17.12, "oan": 184.16}
# Shared-backbone cost of the fused OAN head at tap 10 (raw/audit/backbone_share_output.txt).
FUSED = {"prefix": 101.37, "head": 0.673}
COMPONENTS = REPO / "reports/r1/speed/components.json"
if COMPONENTS.exists():
    _c = json.load(open(COMPONENTS))
    GATE_GFLOPS.update({k: _c["gates"][k]["gflops"] for k in BB})
    GATE_GFLOPS["sweep_mbv3large"] = _c["gates"]["mbv3large_sweep"]["gflops"]
    DET_GFLOPS.update(multiclass=_c["detectors"]["yolo11m_multiclass"]["gflops"],
                      shipsp50=_c["detectors"]["yolo11m_ships_p50"]["gflops"],
                      yolo11n=_c["detectors"]["yolo11n_ships"]["gflops"],
                      oan=_c["detectors"]["yolo11m_oan_joint"]["gflops"])
    FUSED.update(prefix=_c["prefix"]["10"]["prefix_gflops"], head=_c["prefix"]["10"]["head_gflops"])
# Selection metric: name in evaluate(), index in _score_candidate's result, index in BootstrapMAP's.
METRICS = {"coco": ("mAP50_coco", 6, 1), "voc07": ("mAP50_voc07", 5, 0)}
SEL = {"metric": METRICS["coco"]}


def task_classes(task: str) -> list[int]:
    return list(range(16)) if task == "any" else [CLASS_ID[task]]


def gate_files(preset: str, task: str) -> dict[str, tuple[str, float]]:
    slug = task.replace(" ", "_")
    if preset == "sweep":
        return {"mbv3large": (f"reports/gate_sweep_scores/gate_sweep_{slug}_val.jsonl", GATE_GFLOPS["sweep_mbv3large"])}
    if preset in ("pillars", "yolo11n", "shipsp50"):
        sub = {"ship": "reports/gate_scores/gate_{bb}_val.jsonl",
               "plane": "reports/planes/gate_scores/gate_planes_{bb}_val.jsonl",
               "small vehicle": "reports/small_vehicle/gate_scores/gate_small_vehicle_{bb}_val.jsonl"}[task]
        return {bb: (sub.format(bb=bb), GATE_GFLOPS[bb]) for bb in BB}
    raise ValueError(preset)


def load_scores(path: str | Path, tiles: list[str]) -> np.ndarray:
    prob = {}
    for line in open(REPO / path):
        r = json.loads(line)
        prob[r["tile_id"]] = float(r["prob"])
    missing = [t for t in tiles if t not in prob]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} tiles without a score, e.g. {missing[:3]}")
    return np.array([prob[t] for t in tiles])


def tile_info(tiles: list[str], classes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(positive label for the task, sensor-source bucket) per tile."""
    names = {CLASS_ID_INV[c] for c in classes}
    lab, src = {}, {}
    for line in open(META):
        r = json.loads(line)
        if r["split"] != "val":
            continue
        lab[r["tile_id"]] = int(bool(names & set(r.get("class_counts", {}))))
        src[r["tile_id"]] = r.get("imagesource") or "unknown"
    return np.array([lab[t] for t in tiles]), np.array([src[t] for t in tiles])


CLASS_ID_INV = {v: k for k, v in CLASS_ID.items()}


class Calibrator:
    """A calibration map fitted on one fold, applicable to any tiles."""

    def __init__(self, name: str, probs: np.ndarray, labels: np.ndarray, buckets: np.ndarray):
        self.name = name
        if name == "identity":
            self.f = None
        elif name == "context_adaptive":
            ca = fit_context_adaptive(list(range(len(probs))), probs, labels,
                                      bucket_of=lambda i: buckets[i], target_recall=0.99)
            self.thr = dict(ca.thresholds)
            self.fallback = ca.fallback
        else:
            self.f = build_calibration(name).fit(probs, labels)

    def __call__(self, probs: np.ndarray, buckets: np.ndarray) -> np.ndarray:
        if self.name == "identity":
            return probs
        if self.name == "context_adaptive":
            shift = np.array([0.5 - self.thr.get(b, self.fallback) for b in buckets])
            return np.clip(probs + shift, 0.0, 1.0)
        return self.f.transform(probs)


# Globals shared with worker processes (fork).
G: dict = {}


def _score_candidate(args):
    gi, ci, thr, fold = args
    s = G["cal_scores"][gi][ci][fold]          # calibrated scores of all tiles, fitted on `fold`
    acc = s >= thr
    tmask = G["fold_tiles"][fold]
    a = acc[tmask].mean()
    m = evaluate(G["cache"], acc & tmask, G["classes"], images=G["fold_images"][fold])
    return gi, ci, thr, fold, float(a), m["mAP50_voc07"], m["mAP50_coco"]


def cost(gate_g: float, accept: float, model: str, det_g: float) -> float:
    if model == "fused":
        p = FUSED["prefix"] / det_g
        return p + FUSED["head"] / det_g + accept * (1 - p)
    return gate_g / det_g + accept


def full_metrics(cache: EvalCache, acc: np.ndarray, classes: list[int], labels: np.ndarray,
                 images: np.ndarray | None = None, tiles: np.ndarray | None = None) -> dict:
    """Image-level AP plus tile statistics. Tiles labelled -1 (class unknown, e.g. SODA-A ignore
    regions) count towards the accept rate but not towards TPR, FPR or p+."""
    tmask = np.ones(len(acc), bool) if tiles is None else tiles
    m = evaluate(cache, acc & tmask, classes, images=images, iou_thresholds=IOU_THRESHOLDS)
    a = acc[tmask]
    lab = labels[tmask]
    return dict(m, accept_rate=float(a.mean()), tpr=float(a[lab == 1].mean()) if (lab == 1).any() else float("nan"),
                fpr=float(a[lab == 0].mean()) if (lab == 0).any() else float("nan"),
                p_plus=float((lab[lab >= 0] == 1).mean()), n_tiles=int(tmask.sum()),
                n_unlabelled=int((lab < 0).sum()))


def bootstrap(cache: EvalCache, acc: np.ndarray, classes: list[int], gate_g: float, det_g: float,
              model: str, n: int = 1000, seed: int = 67) -> dict:
    """Cluster bootstrap over images: mAP change and saving of `acc` against the full pass."""
    rng = np.random.default_rng(seed)
    full = np.ones(len(acc), bool)
    bf = BootstrapMAP(cache, match(cache, merged_keep(cache, full, classes=classes), 0.5, classes), classes)
    bc = BootstrapMAP(cache, match(cache, merged_keep(cache, acc, classes=classes), 0.5, classes), classes)
    n_img = len(cache.images)
    d_voc, d_coco, save = [], [], []
    for _ in range(n):
        w = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img).astype(float)
        fv, fc = bf(w)
        cv, cc = bc(w)
        tw = w[cache.tile_image]
        a = float((tw * acc).sum() / tw.sum())
        d_voc.append(cv - fv); d_coco.append(cc - fc); save.append(1 - cost(gate_g, a, model, det_g))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"n": n, "delta_mAP50_voc07_ci95": q(d_voc), "delta_mAP50_coco_ci95": q(d_coco),
            "saving_ci95": q(save)}


def risk_lower_bound(cache: EvalCache, acc: np.ndarray, classes: list[int], images: np.ndarray,
                     n: int = 200, q: float = 5.0, seed: int = 67) -> float:
    """Lower q-th percentile of the paired-bootstrap mAP@0.5 change (selection metric)
    of `acc` against the full pass, resampling the images in `images`."""
    rng = np.random.default_rng(seed)
    full = images[cache.tile_image]
    bf = BootstrapMAP(cache, match(cache, merged_keep(cache, full, classes=classes, images=images),
                                   0.5, classes, images), classes)
    bc = BootstrapMAP(cache, match(cache, merged_keep(cache, acc & full, classes=classes, images=images),
                                   0.5, classes, images), classes)
    idx = np.nonzero(images)[0]
    d = []
    for _ in range(n):
        w = np.zeros(len(images))
        np.add.at(w, rng.choice(idx, len(idx)), 1.0)
        bi = SEL["metric"][2]
        d.append(bc(w)[bi] - bf(w)[bi])
    return float(np.percentile(d, q))


def run_task(preset: str, task: str, cache_path: str, det_key: str, model: str, budget: float,
             n_quantiles: int, workers: int, gates: dict[str, tuple[str, float]] | None = None,
             transfer: dict | None = None, out_name: str | None = None,
             det_gflops: float | None = None) -> dict:
    t0 = time.time()
    cache = EvalCache.load(REPO / cache_path)
    classes = task_classes(task) if det_key != "hrsc" else [0]
    labels, buckets = tile_info(cache.tiles, classes)
    folds = yaml.safe_load(open(FOLD_FILE))
    img_fold = {im: f for f in ("A", "B") for im in folds[f]}
    fold_images = {f: np.array([img_fold[im] == f for im in cache.images]) for f in ("A", "B")}
    fold_images["all"] = np.ones(len(cache.images), bool)
    fold_tiles = {f: fold_images[f][cache.tile_image] for f in fold_images}
    gates = gates or gate_files(PRESETS[preset].get("gate_preset", preset), task)
    det_g = det_gflops or DET_GFLOPS[det_key]
    names = list(gates)
    raw = {g: load_scores(gates[g][0], cache.tiles) for g in names}

    # calibrators fitted on each fold (and on all of val for the same-split comparison)
    cal_scores, calibrators = [], []
    for g in names:
        per_c, per_cal = [], []
        for c in CALIBS:
            d, dc = {}, {}
            for f in ("A", "B", "all"):
                tm = fold_tiles[f]
                cal = Calibrator(c, raw[g][tm], labels[tm], buckets[tm])
                dc[f] = cal
                d[f] = cal(raw[g], buckets)
            per_c.append(d); per_cal.append(dc)
        cal_scores.append(per_c); calibrators.append(per_cal)

    G.update(cache=cache, classes=classes, cal_scores=cal_scores, fold_tiles=fold_tiles,
             fold_images=fold_images)
    jobs = []
    thr_grid = {}
    for gi in range(len(names)):
        for ci in range(len(CALIBS)):
            for f in ("A", "B", "all"):
                s = cal_scores[gi][ci][f][fold_tiles[f]]
                thr = np.unique(np.concatenate([[0.0], np.quantile(s, np.linspace(0, 1, n_quantiles + 1))]))
                thr_grid[(gi, ci, f)] = thr
                jobs += [(gi, ci, float(t), f) for t in thr]
    with Pool(workers) as pool:
        res = pool.map(_score_candidate, jobs, chunksize=8)

    full = {f: evaluate(cache, fold_tiles[f], classes, images=fold_images[f]) for f in ("A", "B", "all")}
    # Three selection rules. "risk" is the headline: it fixes the backbone by validation
    # PR-AUC and the calibration to identity (monotone maps do not move the frontier), and
    # chooses only the threshold, as the cheapest one whose paired-bootstrap 5th-percentile
    # mAP change on the selection fold stays within the budget. The two "full" rules search
    # every backbone, calibration map and threshold; they overfit the selection fold (picking
    # the cheapest of thousands of candidates), which is reported as a finding.
    from sklearn.metrics import average_precision_score
    selected = {"risk": {}, "risk_full": {}, "plain_full": {}}
    prauc = {}
    for f in ("A", "B", "all"):
        tm = fold_tiles[f]
        prauc[f] = [float(average_precision_score(labels[tm], raw[g][tm])) for g in names]
        best_gi = int(np.argmax(prauc[f]))
        id_ci = CALIBS.index("identity")
        mkey, midx, _ = SEL["metric"]
        floor = full[f][mkey] - budget
        def as_sel(r, extra=None):
            d = {"gate": names[r[0]], "calibration": CALIBS[r[1]], "threshold": r[2],
                 "select_accept": r[4], "select_mAP50_voc07": r[5], "select_mAP50_coco": r[6],
                 "select_full_mAP50_voc07": full[f]["mAP50_voc07"],
                 "select_full_mAP50_coco": full[f]["mAP50_coco"], "gi": r[0], "ci": r[1],
                 "select_fold_prauc": prauc[f][r[0]]}
            return dict(d, **(extra or {}))
        def risk_pick(cands):
            for k, r in enumerate(cands):
                acc = cal_scores[r[0]][r[1]][f] >= r[2]
                lb = risk_lower_bound(cache, acc, classes, fold_images[f])
                if lb >= -budget:
                    return as_sel(r, {"bootstrap_lower_bound": lb, "candidates_checked": k + 1})
            r = max(cands, key=lambda r: r[4])   # nothing passes: the most permissive candidate
            return as_sel(r, {"bootstrap_lower_bound": None, "candidates_checked": len(cands)})
        key = lambda r: (cost(gates[names[r[0]]][1], r[4], model, det_g), -r[midx])
        allc = sorted((r for r in res if r[3] == f and r[midx] >= floor), key=key)
        restricted = [r for r in allc if r[0] == best_gi and r[1] == id_ci]
        selected["plain_full"][f] = as_sel(allc[0])
        selected["risk_full"][f] = risk_pick(allc)
        selected["risk"][f] = risk_pick(restricted)

    def decide(sel: dict, fitted_on: str) -> np.ndarray:
        return cal_scores[sel["gi"]][sel["ci"]][fitted_on] >= sel["threshold"]

    out = {"preset": preset, "task": task, "cache": cache_path, "detector_gflops": det_g,
           "fused_cost": dict(FUSED) if model == "fused" else None,
           "cost_model": model, "budget_pp": budget * 100, "selection_metric": SEL["metric"][0],
           "n_quantiles": n_quantiles,
           "classes": classes, "gates": {g: {"scores": gates[g][0], "gflops": gates[g][1]} for g in names},
           "protocol": "select on one fold, report on the other, pool; budget on image-level "
                       f"{SEL['metric'][0]} (DOTA devkit matching, difficult objects ignored)",
           "rules": {"risk": "backbone with the highest PR-AUC on the selection fold, identity calibration, "
                             "cheapest threshold whose paired-bootstrap 5th-percentile mAP@0.5 change on the "
                             "selection fold is within the budget (200 image resamples); headline",
                     "risk_full": "same bootstrap rule searched over every backbone, calibration map and threshold",
                     "plain_full": "cheapest candidate over every backbone, calibration and threshold whose "
                                   "point estimate on the selection fold is within the budget"}}
    masks = {}
    fm = full_metrics(cache, np.ones(len(cache.tiles), bool), classes, labels)
    out["full_pass"] = fm
    for rule in ("risk", "risk_full", "plain_full"):
        res_rule = {"folds": {}}
        for sel_f, rep_f in (("A", "B"), ("B", "A")):
            sel = selected[rule][sel_f]
            acc = decide(sel, sel_f)
            masks[f"{rule}_report_{rep_f}"] = acc & fold_tiles[rep_f]
            rep = full_metrics(cache, acc, classes, labels, images=fold_images[rep_f], tiles=fold_tiles[rep_f])
            rep_full = full_metrics(cache, np.ones(len(acc), bool), classes, labels,
                                    images=fold_images[rep_f], tiles=fold_tiles[rep_f])
            res_rule["folds"][f"select_{sel_f}_report_{rep_f}"] = {
                "selected": {k: v for k, v in sel.items() if k not in ("gi", "ci")},
                "report": rep, "report_full_pass": rep_full,
                "report_saving": 1 - cost(gates[sel["gate"]][1], rep["accept_rate"], model, det_g),
                "report_delta_mAP50_voc07": rep["mAP50_voc07"] - rep_full["mAP50_voc07"],
                "report_delta_mAP50_coco": rep["mAP50_coco"] - rep_full["mAP50_coco"]}
        pooled = masks[f"{rule}_report_A"] | masks[f"{rule}_report_B"]
        masks[f"{rule}_pooled"] = pooled
        gA = gates[selected[rule]["B"]["gate"]][1]   # fold-A tiles are decided by fold B's choice
        gB = gates[selected[rule]["A"]["gate"]][1]
        g_pool = (gA * fold_tiles["A"].sum() + gB * fold_tiles["B"].sum()) / len(pooled)
        pm = full_metrics(cache, pooled, classes, labels)
        s = 1 - cost(g_pool, pm["accept_rate"], model, det_g)
        res_rule["pooled"] = dict(pm, gate_gflops=g_pool, saving=s,
                                  delta_mAP50_voc07=pm["mAP50_voc07"] - fm["mAP50_voc07"],
                                  delta_mAP50_coco=pm["mAP50_coco"] - fm["mAP50_coco"],
                                  delta_mAP50_95_coco=pm["mAP50_95_coco"] - fm["mAP50_95_coco"],
                                  saving_if_gate_free=1 - pm["accept_rate"],
                                  charged_share=float(pm["p_plus"] * (1 - pm["tpr"]) / s) if s > 0 else None,
                                  bootstrap=bootstrap(cache, pooled, classes, g_pool, det_g, model))
        out[rule] = res_rule
    out["pooled"] = out["risk"]["pooled"]      # headline rule
    # same-split (old protocol, image level) for comparison
    sel = selected["plain_full"]["all"]
    acc_all = decide(sel, "all")
    masks["same_split"] = acc_all
    sm = full_metrics(cache, acc_all, classes, labels)
    out["same_split"] = dict(sm, selected={k: v for k, v in sel.items() if k not in ("gi", "ci")},
                             saving=1 - cost(gates[sel["gate"]][1], sm["accept_rate"], model, det_g),
                             delta_mAP50_voc07=sm["mAP50_voc07"] - fm["mAP50_voc07"],
                             delta_mAP50_coco=sm["mAP50_coco"] - fm["mAP50_coco"])
    # sweep curves on all of val, identity calibration, per gate (for figures)
    out["curves"] = {}
    ci = CALIBS.index("identity")
    for gi, g in enumerate(names):
        rows = [r for r in res if r[0] == gi and r[1] == ci and r[3] == "all"]
        rows.sort(key=lambda r: -r[4])
        s_all = cal_scores[gi][ci]["all"]
        out["curves"][g] = [{"threshold": r[2], "accept_rate": r[4], "mAP50_voc07": r[5], "mAP50_coco": r[6],
                             "tpr": float((s_all >= r[2])[labels == 1].mean())} for r in rows]
    # calibration error on held-out folds (fit on one fold, measure on the other)
    from src.calibration import expected_calibration_error
    out["ece_heldout"] = {}
    for gi, g in enumerate(names):
        e = {}
        for ci, c in enumerate(CALIBS):
            vals = []
            for fit_f, ev_f in (("A", "B"), ("B", "A")):
                tm = fold_tiles[ev_f]
                vals.append(expected_calibration_error(cal_scores[gi][ci][fit_f][tm], labels[tm])["ece"])
            same = expected_calibration_error(cal_scores[gi][ci]["all"], labels)["ece"]
            e[c] = {"heldout_mean": float(np.mean(vals)), "heldout_folds": [float(v) for v in vals],
                    "same_split": float(same)}
        out["ece_heldout"][g] = e
    out["selections"] = {rule: {f: {k: v for k, v in s.items() if k not in ("gi", "ci")}
                                for f, s in per_f.items()} for rule, per_f in selected.items()}
    if transfer:
        out["transfer"] = transfer_eval(transfer, selected, calibrators, names, gates, model, det_g, task)
    out["seconds"] = round(time.time() - t0, 1)
    dest = REPO / "reports/r1/cascade" / preset
    dest.mkdir(parents=True, exist_ok=True)
    slug = out_name or task.replace(" ", "_")
    (dest / f"{slug}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(dest / f"{slug}_masks.npz", tiles=np.array(cache.tiles), **masks)
    return out


def transfer_eval(tr: dict, selected: dict, calibrators: list, names: list, gates: dict,
                  model: str, det_g: float, task: str) -> dict:
    """Apply operating points chosen on all of DOTA val to another dataset, untouched by selection."""
    from scipy.stats import beta
    cache = EvalCache.load(REPO / tr["cache"])
    slug = task.replace(" ", "_")
    classes = task_classes(task) if tr["classes"] == "task" else tr["classes"]
    want = {CLASS_ID_INV[c] for c in classes}
    lab_of = {}
    for line in open(REPO / tr["meta"]):
        r = json.loads(line)
        if "is_positive" in r:
            lab_of[r["tile_id"]] = int(r["is_positive"])
        else:   # multi-class metadata: positive for the task's classes; -1 where the class is unknown
            lab_of[r["tile_id"]] = -1 if r.get("has_ignore") else int(bool(want & set(r.get("class_counts", {}))))
    labels = np.array([lab_of[t] for t in cache.tiles])
    buckets = np.array([tr.get("bucket", "transfer")] * len(cache.tiles))
    full = full_metrics(cache, np.ones(len(cache.tiles), bool), classes, labels)
    out = {"cache": tr["cache"], "full_pass": full, "note": tr.get("note", "")}
    for rule in ("risk", "plain_full"):
        sel = selected[rule]["all"]
        gname = names[sel["gi"]]
        raw = load_scores(tr["scores"][gname].format(slug=slug, bb=gname), cache.tiles)
        acc = calibrators[sel["gi"]][sel["ci"]]["all"](raw, buckets) >= sel["threshold"]
        m = full_metrics(cache, acc, classes, labels)
        neg = labels == 0
        k, n = int(acc[neg].sum()), int(neg.sum())
        ci = [float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0,
              float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]
        s = 1 - cost(gates[gname][1], m["accept_rate"], model, det_g)
        out[rule] = dict(m, selected={kk: v for kk, v in sel.items() if kk not in ("gi", "ci")},
                         saving=s, delta_mAP50_voc07=m["mAP50_voc07"] - full["mAP50_voc07"],
                         delta_mAP50_coco=m["mAP50_coco"] - full["mAP50_coco"],
                         fpr_counts=[k, n], fpr_ci95_clopper_pearson=ci,
                         charged_share=float(m["p_plus"] * (1 - m["tpr"]) / s) if s > 0 else None)
    return out


PRESETS = {
    "sweep": dict(cache="runs/r1_cache/multiclass_val.npz", det="multiclass", model="decoupled",
                  tasks=None),   # read from reports/gate_tasks.json
    "pillars": dict(cache="runs/r1_cache/multiclass_val.npz", det="multiclass", model="decoupled",
                    tasks=["ship", "plane", "small vehicle"]),
    "shipsp50": dict(cache="runs/r1_cache/shipsp50_val.npz", det="shipsp50", model="decoupled", tasks=["ship"]),
    "yolo11n": dict(cache="runs/r1_cache/yolo11n_ships_1024_val.npz", det="yolo11n", model="decoupled",
                    tasks=["ship"]),
    # fused OAN head (shared-backbone cost) against its matched independent gate, seed 67
    "fused": dict(cache="runs/r1_cache/oan_val.npz", det="oan", model="fused", tasks=["ship"],
                  gates={"oan_head": ("reports/oan/gate_oan_ships_val.jsonl", 0.0)}),
    "indep": dict(cache="runs/r1_cache/shipsp50_val.npz", det="shipsp50", model="decoupled", tasks=["ship"],
                  gates={"mbv3large": ("reports/gate_scores/gate_mbv3large_val.jsonl", GATE_GFLOPS["mbv3large"])}),
    # HRSC2016: operating point chosen on DOTA val only, then applied unchanged
    "hrsc": dict(cache="runs/r1_cache/shipsp50_val.npz", det="shipsp50", model="decoupled", tasks=["ship"],
                 gates={"mbv3large": ("reports/gate_scores/gate_mbv3large_val.jsonl", GATE_GFLOPS["mbv3large"])},
                 transfer={"cache": "runs/r1_cache/shipsp50_hrsc.npz",
                           "meta": "data/processed/hrsc2016/metadata/tiles.jsonl", "classes": [0],
                           "scores": {"mbv3large": "reports/hrsc/gate_mbv3large_test.jsonl"},
                           "bucket": "HRSC2016",
                           "note": "untiled HRSC2016 test images; calibration and threshold fitted on DOTA val; "
                                   "the context-adaptive map falls back to its global threshold for this source"}),
    # SODA-A validation (independent annotation team and protocol): operating points chosen on DOTA val
    "soda_sweep": dict(cache="runs/r1_cache/multiclass_val.npz", det="multiclass", model="decoupled",
                       tasks=["ship", "plane", "small vehicle"], gate_preset="sweep",
                       transfer={"cache": "runs/r1_cache/multiclass_soda.npz",
                                 "meta": "data/processed/soda_a/metadata/tiles.jsonl", "classes": "task",
                                 "scores": {"mbv3large": "reports/r1/soda/gate_scores/sweep_{slug}.jsonl"},
                                 "bucket": "SODA-A",
                                 "note": "SODA-A val tiled 1024/200; detections in ignore regions dropped; "
                                         "tiles touching an ignore region excluded from p+, TPR and FPR"}),
    "soda_pillars": dict(cache="runs/r1_cache/multiclass_val.npz", det="multiclass", model="decoupled",
                         tasks=["ship", "plane", "small vehicle"], gate_preset="pillars",
                         transfer={"cache": "runs/r1_cache/multiclass_soda.npz",
                                   "meta": "data/processed/soda_a/metadata/tiles.jsonl", "classes": "task",
                                   "scores": {bb: "reports/r1/soda/gate_scores/{slug}_{bb}.jsonl" for bb in BB},
                                   "bucket": "SODA-A",
                                   "note": "SODA-A val tiled 1024/200; detections in ignore regions dropped; "
                                           "tiles touching an ignore region excluded from p+, TPR and FPR"}),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--budget", type=float, default=0.03, help="mAP@0.5 tolerance, absolute")
    p.add_argument("--metric", choices=sorted(METRICS), default="coco",
                   help="AP interpolation the budget applies to: 101-point (coco) or 11-point (voc07)")
    p.add_argument("--quantiles", type=int, default=200)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--cache", default=None, help="override the preset's cache")
    p.add_argument("--gate", action="append", default=None,
                   help="override gates: NAME=PATH[:GFLOPS] (repeatable)")
    p.add_argument("--out-name", default=None, help="output file stem (default: task)")
    p.add_argument("--det-gflops", type=float, default=None,
                   help="override the detector's GFLOPs per tile (e.g. a reduced input size)")
    p.add_argument("--fused-prefix", type=float, default=None,
                   help="fused cost model: GFLOPs of the backbone prefix up to the tap (default layer 10)")
    p.add_argument("--fused-head", type=float, default=None, help="fused cost model: GFLOPs of the gate head")
    args = p.parse_args()
    SEL["metric"] = METRICS[args.metric]
    if args.fused_prefix is not None:
        FUSED["prefix"] = args.fused_prefix
    if args.fused_head is not None:
        FUSED["head"] = args.fused_head
    cfg = dict(PRESETS[args.preset])
    if args.cache:
        cfg["cache"] = args.cache
    if args.gate:
        cfg["gates"] = {}
        for g in args.gate:
            name, rest = g.split("=", 1)
            path, _, gf = rest.partition(":")
            cfg["gates"][name] = (path, float(gf) if gf else GATE_GFLOPS.get(name, 0.0))
    tasks = args.tasks or cfg["tasks"] or [t["task"] for t in json.load(open(REPO / "reports/gate_tasks.json"))]
    for task in tasks:
        out = run_task(args.preset, task, cfg["cache"], cfg["det"], cfg["model"], args.budget,
                       args.quantiles, args.workers, gates=cfg.get("gates"), transfer=cfg.get("transfer"),
                       out_name=args.out_name, det_gflops=args.det_gflops)
        rk, rf, pf, ss = (out["risk"]["pooled"], out["risk_full"]["pooled"], out["plain_full"]["pooled"],
                          out["same_split"])
        fmt = lambda x: (f"{100*x['saving']:.1f}%/{100*x['delta_mAP50_coco']:+.2f}pp"
                         f"(voc {100*x['delta_mAP50_voc07']:+.2f})")
        print(f"[{args.preset}] {task:<20} p+={rk['p_plus']:.3f} | risk {fmt(rk)} TPR={rk['tpr']:.3f} | "
              f"risk_full {fmt(rf)} | plain_full {fmt(pf)} | same-split {fmt(ss)} ({out['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
