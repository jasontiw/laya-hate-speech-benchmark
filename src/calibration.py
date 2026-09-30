"""Calibration measurement for probability-producing models.

Two quantities, both for the **binary** hate score (P(hate speech)):

* **Brier score** - mean squared error of the probability; lower is better.
* **ECE** (expected calibration error) - the gap between predicted probability and
  observed frequency, averaged over confidence bins; lower is better.

This module *measures* calibration. It does not fit anything: temperature fitting
(or any other recalibration) must be done on a validation split, never on the test
set. Laya's own documentation makes the same point, and its shipped checkpoints are
over-confident as published.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _binary_gold(y_true: Sequence[str], positive: str) -> np.ndarray:
    return np.asarray([1 if str(value) == positive else 0 for value in y_true], dtype=int)


def brier_score(y_true: Sequence[str], scores: Sequence[float], positive: str) -> Optional[float]:
    gold = _binary_gold(y_true, positive)
    if gold.size == 0:
        return None
    probabilities = np.asarray(scores, dtype=float)
    return float(np.mean((probabilities - gold) ** 2))


def reliability_bins(
    y_true: Sequence[str],
    scores: Sequence[float],
    positive: str,
    bins: int = 10,
) -> Dict[str, Any]:
    """Bin the predicted probability and compare it with the observed frequency.

    Returns the bin edges, the mean predicted probability and the observed
    frequency per bin, plus the counts. Empty bins are omitted.
    """
    gold = _binary_gold(y_true, positive)
    probabilities = np.asarray(scores, dtype=float)
    if gold.size == 0:
        return {"bins": [], "ece": None}

    edges = np.linspace(0.0, 1.0, bins + 1)
    # Right-closed bins so a probability of exactly 1.0 lands in the last bin.
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)

    rows: List[Dict[str, Any]] = []
    total = len(gold)
    ece = 0.0
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(probabilities[mask].mean())
        observed = float(gold[mask].mean())
        ece += (count / total) * abs(observed - mean_predicted)
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "mean_predicted": mean_predicted,
                "observed_frequency": observed,
                "gap": observed - mean_predicted,
            }
        )
    return {"bins": rows, "ece": float(ece), "n": total}


def calibration_report(
    y_true: Sequence[str],
    scores: Optional[Sequence[float]],
    positive: str = "hate speech",
    bins: int = 10,
) -> Optional[Dict[str, Any]]:
    """Brier + ECE + reliability bins for one model, or None without a score."""
    if scores is None:
        return None
    gold = _binary_gold(y_true, positive)
    if gold.sum() == 0 or gold.sum() == len(gold):
        return None
    reliability = reliability_bins(y_true, scores, positive, bins=bins)
    return {
        "positive": positive,
        "bins": bins,
        "brier": brier_score(y_true, scores, positive),
        "ece": reliability["ece"],
        "reliability": reliability["bins"],
        "base_rate": float(gold.mean()),
    }
