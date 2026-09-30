"""Post-hoc calibration of the hate score itself (Platt scaling / isotonic regression).

Laya's own temperature map calibrates the confidence of the answer it picked, which the
benchmark measured to be a *different* target (see the report's recalibration table): the
answer-confidence ECE improved a lot while the hate-score ECE barely moved.

This module calibrates the quantity a detector actually thresholds: ``P(hate speech)``.

Both maps are strictly monotone, so they cannot change a ranking, a prediction, or
PR-AUC. What changes is whether the number means what it says. That distinction is the
whole point: this improves *trust*, not *detection*.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import HATE_LABEL

EPS = 1e-6
METHODS = ("platt", "isotonic")


def _clip(scores: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clamped = np.clip(probabilities, EPS, 1.0 - EPS)
    return np.log(clamped / (1.0 - clamped))


def binary_targets(gold_labels: Sequence[str], positive: str = HATE_LABEL) -> np.ndarray:
    return np.asarray([1 if str(value) == positive else 0 for value in gold_labels], dtype=int)


def fit(scores: Sequence[float], targets: Sequence[int], method: str) -> Dict[str, Any]:
    """Fit a monotone map, returning JSON-serialisable parameters."""
    probabilities = _clip(scores)
    y = np.asarray(targets, dtype=int)
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        # A very large C means "no regularisation": Platt scaling is a two-parameter fit.
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(_logit(probabilities).reshape(-1, 1), y)
        return {"method": "platt", "a": float(model.coef_[0][0]), "b": float(model.intercept_[0])}
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(probabilities, y)
        return {
            "method": "isotonic",
            "x": [float(v) for v in model.X_thresholds_],
            "y": [float(v) for v in model.y_thresholds_],
        }
    raise ValueError("unknown calibration method %r" % method)


def apply(params: Dict[str, Any], scores: Sequence[float]) -> np.ndarray:
    """Apply a fitted map. Kept dependency-free so a fitted map is portable JSON."""
    probabilities = _clip(scores)
    if params.get("method") == "platt":
        logits = params["a"] * _logit(probabilities) + params["b"]
        return 1.0 / (1.0 + np.exp(-logits))
    x = np.asarray(params.get("x", []), dtype=float)
    y = np.asarray(params.get("y", []), dtype=float)
    if x.size == 0:
        return probabilities
    return np.interp(probabilities, x, y, left=float(y[0]), right=float(y[-1]))


def _numbers(gold_labels: Sequence[str], scores: Sequence[float], bins: int) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score

    from .calibration import brier_score, reliability_bins

    targets = binary_targets(gold_labels)
    clipped = _clip(scores)
    average_precision = None
    if 0 < int(targets.sum()) < len(targets):
        average_precision = float(average_precision_score(targets, clipped))
    return {
        "brier": brier_score(gold_labels, scores, HATE_LABEL),
        "ece": (reliability_bins(gold_labels, scores, HATE_LABEL, bins=bins) or {}).get("ece"),
        # Reported so the report can *verify* the monotonicity claim instead of asserting it:
        # Platt is strictly monotone and leaves both numbers untouched; isotonic merges
        # scores into blocks, so ties appear and `average_precision` can fall.
        "average_precision": average_precision,
        "unique_scores": int(np.unique(clipped).size),
    }


def compare(
    validation_scores: Sequence[float],
    validation_gold: Sequence[str],
    test_scores: Sequence[float],
    test_gold: Sequence[str],
    bins: int = 10,
    methods: Sequence[str] = METHODS,
) -> Dict[str, Any]:
    """Fit on validation, measure on test, for every requested method."""
    targets = binary_targets(validation_gold)
    before = _numbers(test_gold, test_scores, bins)
    baseline_decisions = _clip(test_scores) >= 0.5

    results: Dict[str, Any] = {
        "fitted_on": "validation",
        "fitted_rows": int(len(targets)),
        "test_before": before,
    }
    for method in methods:
        params = fit(validation_scores, targets, method)
        calibrated = apply(params, test_scores)
        changed = int(((_clip(calibrated) >= 0.5) != baseline_decisions).sum())
        results[method] = {
            "params": params,
            "test_after": _numbers(test_gold, calibrated, bins),
            "hate_decisions_changed_at_0_5": changed,
        }
    return results
