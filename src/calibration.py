"""Calibration and threshold-selection methods (Pillar 2).

Six methods, in order of sophistication:

1. naive 0.5
2. recall-targeted: pick the threshold on validation that achieves a target
   recall on positive tiles (e.g. 0.99)
3. mAP-optimal grid: sweep thresholds, evaluate end-to-end cascade mAP at each,
   keep the threshold with highest mAP — the *correct* metric per research
   plan §7.2 / §15.7. Implemented here as an iterator: callers supply a
   downstream-mAP function ``score_fn(threshold) -> mAP`` so we don't couple
   calibration to detector internals.
4. score-calibrated then thresholded: temperature scaling, Platt scaling,
   isotonic regression. All three transform raw probabilities first; threshold
   is applied to the transformed scores.

Methods 5 (context-adaptive) and 6 (learned thresholds) live in this file too,
but as Phase-3 stubs to flesh out once Pillar-1 baselines are in place.

The classifier should be trained with BCE/focal — calibration *adjusts* the
output distribution post-hoc. It does not retrain the gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


# ---------- Calibration map (probabilities -> probabilities) ---------------


class CalibrationMap:
    """Abstract: given raw gate probs, return calibrated probs in [0, 1]."""

    name: str = "abstract"

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "CalibrationMap":
        raise NotImplementedError

    def transform(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(probs, labels).transform(probs)


class IdentityCalibration(CalibrationMap):
    name = "identity"

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IdentityCalibration":
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return probs


class TemperatureScaling(CalibrationMap):
    """Single scalar T applied to logits (recovered from probs via the logit
    function). Optimizes NLL on the validation set with a 1-D LBFGS step. T<1
    sharpens, T>1 softens.
    """

    name = "temperature"

    def __init__(self) -> None:
        self.T: float = 1.0

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        import torch

        logits = torch.tensor(self._logit(probs), dtype=torch.float32)
        target = torch.tensor(labels.astype(np.float32))
        T = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / T.clamp(min=1e-3), target)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.T = float(T.detach().clamp(min=1e-3).item())
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self._sigmoid(self._logit(probs) / self.T)


class PlattScaling(CalibrationMap):
    """Logistic regression on logits — the classical Platt method."""

    name = "platt"

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "PlattScaling":
        from sklearn.linear_model import LogisticRegression

        x = TemperatureScaling._logit(probs).reshape(-1, 1)
        clf = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        clf.fit(x, labels)
        self.a = float(clf.coef_.ravel()[0])
        self.b = float(clf.intercept_.ravel()[0])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        x = TemperatureScaling._logit(probs)
        return TemperatureScaling._sigmoid(self.a * x + self.b)


class IsotonicCalibration(CalibrationMap):
    """Non-parametric isotonic regression. Best when the miscalibration shape
    is non-monotone; can overfit on small validation sets — pair with a
    held-out test split.
    """

    name = "isotonic"

    def __init__(self) -> None:
        self._ir = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibration":
        from sklearn.isotonic import IsotonicRegression

        self._ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._ir.fit(probs, labels)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self._ir is None:
            raise RuntimeError("IsotonicCalibration.fit must be called first")
        return np.asarray(self._ir.transform(probs))


CALIBRATION_REGISTRY: dict[str, type[CalibrationMap]] = {
    "identity": IdentityCalibration,
    "temperature": TemperatureScaling,
    "platt": PlattScaling,
    "isotonic": IsotonicCalibration,
}


def build_calibration(name: str) -> CalibrationMap:
    if name not in CALIBRATION_REGISTRY:
        raise ValueError(f"Unknown calibration '{name}'. Known: {sorted(CALIBRATION_REGISTRY)}")
    return CALIBRATION_REGISTRY[name]()


# ---------- Calibration metrics --------------------------------------------


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict[str, float]:
    """ECE and MCE on equal-width bins, with the mean abs gap per bin returned
    so callers can plot reliability diagrams without re-binning.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probs, edges[1:-1])
    n = len(probs)
    ece = 0.0
    mce = 0.0
    bins: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = indices == b
        if not mask.any():
            bins.append({"bin": b, "n": 0, "mean_prob": 0.0, "mean_label": 0.0, "abs_gap": 0.0})
            continue
        mean_p = float(probs[mask].mean())
        mean_y = float(labels[mask].mean())
        gap = abs(mean_p - mean_y)
        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)
        bins.append({"bin": b, "n": int(mask.sum()), "mean_prob": mean_p, "mean_label": mean_y, "abs_gap": gap})
    return {"ece": float(ece), "mce": float(mce), "bins": bins, "n_bins": n_bins}


# ---------- Threshold selection --------------------------------------------


@dataclass
class ThresholdResult:
    method: str
    threshold: float
    metadata: dict[str, Any]


def threshold_naive() -> ThresholdResult:
    return ThresholdResult(method="naive_0.5", threshold=0.5, metadata={})


def threshold_recall_targeted(
    probs: np.ndarray, labels: np.ndarray, target_recall: float = 0.99
) -> ThresholdResult:
    """Pick the largest threshold whose recall on positives is >= target.

    Larger thresholds reject more tiles, so for a given recall floor we pick the
    most aggressive threshold (largest filter rate) that still meets the floor.
    """
    pos_idx = labels == 1
    pos_probs = np.sort(probs[pos_idx])
    if len(pos_probs) == 0:
        return ThresholdResult(method=f"recall>={target_recall}", threshold=0.0, metadata={"warning": "no positives"})
    k = max(1, int(np.ceil((1.0 - target_recall) * len(pos_probs))))
    threshold = float(pos_probs[max(k - 1, 0)])
    actual_recall = float((probs[pos_idx] >= threshold).mean())
    return ThresholdResult(
        method=f"recall>={target_recall}",
        threshold=threshold,
        metadata={"actual_recall": actual_recall, "n_positive": int(pos_idx.sum())},
    )


def threshold_map_grid(
    score_fn: Callable[[float], float],
    candidate_thresholds: Iterable[float] | None = None,
    n_thresholds: int = 51,
) -> ThresholdResult:
    """Sweep thresholds and pick the one that maximizes ``score_fn(threshold)``.

    ``score_fn`` should return end-to-end mAP for the cascade at that threshold.
    The caller is responsible for caching detector outputs so each evaluation
    is cheap (the typical pattern: pre-compute detector outputs on every tile,
    then for each threshold simply mask out rejected tiles before computing mAP).
    """
    if candidate_thresholds is None:
        candidate_thresholds = list(np.linspace(0.0, 1.0, n_thresholds))
    candidate_thresholds = list(candidate_thresholds)
    scores = [(t, float(score_fn(t))) for t in candidate_thresholds]
    best_t, best_s = max(scores, key=lambda kv: kv[1])
    return ThresholdResult(
        method="map_grid",
        threshold=float(best_t),
        metadata={"sweep": scores, "best_score": best_s},
    )


# ---------- End-to-end calibration report ----------------------------------


def calibration_report(
    probs: np.ndarray,
    labels: np.ndarray,
    out_dir: str | Path,
    methods: tuple[str, ...] = ("identity", "temperature", "platt", "isotonic"),
    n_bins: int = 15,
) -> dict[str, Any]:
    """Fit every calibrator on the same (probs, labels) and emit a full report:

    - per-method calibrated probs (saved as .npy for fast reload)
    - ECE / MCE with bin-level breakdown
    - JSON summary index

    The probabilities are *unsplit* — caller is responsible for using a held-out
    set. For the cascade workflow we typically fit on val and evaluate
    downstream cascade mAP on test.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"n": int(len(probs)), "positive_rate": float(labels.mean()), "methods": {}}
    for name in methods:
        calib = build_calibration(name).fit(probs, labels)
        calibrated = calib.transform(probs)
        ece = expected_calibration_error(calibrated, labels, n_bins=n_bins)
        np.save(out / f"probs_{name}.npy", calibrated)
        summary["methods"][name] = {
            "ece": ece["ece"],
            "mce": ece["mce"],
            "params": _calibration_params(calib),
        }
        # Save the bin-level reliability data alongside.
        with (out / f"reliability_{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(ece, handle, indent=2)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _calibration_params(calib: CalibrationMap) -> dict[str, Any]:
    if isinstance(calib, TemperatureScaling):
        return {"T": calib.T}
    if isinstance(calib, PlattScaling):
        return {"a": calib.a, "b": calib.b}
    if isinstance(calib, IsotonicCalibration):
        return {"non_parametric": True}
    return {}


# ---------- Method 5: context-adaptive thresholds --------------------------


@dataclass
class ContextAdaptiveThresholds:
    """Per-bucket threshold map keyed on a categorical context (e.g. imagesource
    or GSD bucket). For each bucket independently, sweep thresholds and pick
    the one that maximizes recall subject to a per-bucket filter-rate floor —
    or, when given a downstream score function, maximizes that.

    The model is intentionally cheap: a dict of (bucket -> threshold). At
    inference, look up each tile's bucket and apply the matching threshold.
    """

    thresholds: dict[str, float]
    fallback: float = 0.5

    def threshold_for(self, bucket: str | None) -> float:
        return float(self.thresholds.get(str(bucket), self.fallback))


def fit_context_adaptive(
    tile_ids: list[str],
    probs: np.ndarray,
    labels: np.ndarray,
    bucket_of: Callable[[str], str],
    target_recall: float = 0.99,
    min_bucket_size: int = 50,
    fallback_threshold: float = 0.5,
) -> ContextAdaptiveThresholds:
    """Per-bucket recall-targeted thresholds. ``bucket_of(tile_id) -> bucket_name``.

    Buckets with fewer than ``min_bucket_size`` tiles fall back to the global
    threshold (so we don't pick noisy thresholds on rare contexts).
    """
    buckets: dict[str, list[int]] = {}
    for i, tid in enumerate(tile_ids):
        buckets.setdefault(str(bucket_of(tid)), []).append(i)

    global_threshold = threshold_recall_targeted(probs, labels, target_recall).threshold
    out: dict[str, float] = {}
    for bucket, idx in buckets.items():
        if len(idx) < min_bucket_size:
            out[bucket] = global_threshold
            continue
        sub_probs = probs[idx]
        sub_labels = labels[idx]
        if int(sub_labels.sum()) == 0:
            # All negatives in this bucket — accept nothing (highest threshold).
            out[bucket] = 1.0
            continue
        out[bucket] = threshold_recall_targeted(sub_probs, sub_labels, target_recall).threshold
    return ContextAdaptiveThresholds(thresholds=out, fallback=fallback_threshold or global_threshold)


def apply_context_adaptive(
    tile_ids: list[str],
    probs: np.ndarray,
    bucket_of: Callable[[str], str],
    thresholds: ContextAdaptiveThresholds,
) -> np.ndarray:
    """Return per-tile binary decisions under per-bucket thresholds."""
    decisions = np.zeros(len(tile_ids), dtype=np.int64)
    for i, tid in enumerate(tile_ids):
        t = thresholds.threshold_for(bucket_of(tid))
        decisions[i] = int(probs[i] >= t)
    return decisions


# ---------- Method 6: learned threshold MLP --------------------------------


class LearnedThresholdMLP:
    """Tiny MLP ``(gate_logit, cheap_features) -> threshold`` trained against a
    differentiable surrogate of mAP.

    Implementation note: the cleanest surrogate we have is the per-tile
    contribution to the Pareto rank. A simple proxy that works well in
    practice:

        loss = BCE( sigmoid(gate_score - threshold(x)), is_positive_tile )

    i.e. the MLP learns to pick a threshold that splits positives from
    negatives well. This is *not* end-to-end mAP, but it is a strict
    improvement over the global threshold whenever the optimal threshold is
    feature-conditional. Phase 4 can replace the surrogate with a true mAP
    surrogate (e.g. mAP-attempt loss against frozen detector outputs) if the
    simple version doesn't dominate the global threshold.
    """

    def __init__(self, in_dim: int, hidden: int = 32):
        import torch
        import torch.nn as nn

        self._torch = torch
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def fit(
        self,
        logits: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> "LearnedThresholdMLP":
        torch = self._torch

        x = torch.tensor(np.concatenate([logits[:, None], features], axis=1), dtype=torch.float32)
        y = torch.tensor(labels.astype(np.float32))
        gate = torch.tensor(logits, dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            t = self.model(x).squeeze(-1)
            score = torch.sigmoid(gate - t)
            loss = torch.nn.functional.binary_cross_entropy(score, y)
            loss.backward()
            opt.step()
        return self

    def predict_thresholds(self, logits: np.ndarray, features: np.ndarray) -> np.ndarray:
        torch = self._torch

        x = torch.tensor(np.concatenate([logits[:, None], features], axis=1), dtype=torch.float32)
        with torch.no_grad():
            t = self.model(x).squeeze(-1).numpy()
        return t

    def decisions(
        self,
        logits: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        thresholds = self.predict_thresholds(logits, features)
        # Convert logits back to probs and compare; keeps the contract symmetric
        # with the other thresholding methods.
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= thresholds).astype(np.int64)
