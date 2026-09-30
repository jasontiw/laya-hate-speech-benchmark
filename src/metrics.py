"""Metrics: overall, per-class, hate-specific, ranking and latency.

Covers PRD sections 14, 15, 17 plus the v1 additions:
* hate-vs-rest as a binary problem with PR-AUC (average precision),
* the best achievable operating point on the hate score,
* bootstrap confidence intervals, because a single split is one sample.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from .config import CANONICAL_LABELS, HATE_LABEL, TASK_HATE_BINARY

_LABEL_TO_CODE = {label: i for i, label in enumerate(CANONICAL_LABELS)}
_HATE_CODE = _LABEL_TO_CODE[HATE_LABEL]


# --------------------------------------------------------------------------- #
# Core classification metrics
# --------------------------------------------------------------------------- #


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """All classification metrics for one model, on one shared test set."""
    label_list = list(labels or CANONICAL_LABELS)
    gold = [str(x) for x in y_true]
    pred = [str(x) for x in y_pred]
    n = len(gold)

    accuracy = float(accuracy_score(gold, pred)) if n else 0.0
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=label_list, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, pred, labels=label_list, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        gold, pred, labels=label_list, average="weighted", zero_division=0
    )[2]
    matrix = confusion_matrix(gold, pred, labels=label_list)

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(label_list)
    }

    # Hate speech treated as a binary problem: hate vs everything else.
    hate_index = label_list.index(HATE_LABEL)
    tp = int(matrix[hate_index, hate_index])
    fn = int(matrix[hate_index, :].sum() - tp)
    fp = int(matrix[:, hate_index].sum() - tp)
    tn = int(n - tp - fn - fp)
    hate_precision = tp / (tp + fp) if (tp + fp) else 0.0
    hate_recall = tp / (tp + fn) if (tp + fn) else 0.0
    hate_f1 = (2 * hate_precision * hate_recall / (hate_precision + hate_recall)) if (hate_precision + hate_recall) else 0.0

    return {
        "n": n,
        "labels": label_list,
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "hate": {
            "precision": float(hate_precision),
            "recall": float(hate_recall),
            "f1": float(hate_f1),
            "support": int(support[hate_index]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        },
    }


# --------------------------------------------------------------------------- #
# Hate-vs-rest ranking metrics
# --------------------------------------------------------------------------- #


def hate_score_array(probabilities: Sequence[Optional[Dict[str, float]]]) -> np.ndarray:
    """P(hate speech) for every row; 0.0 where a row has no probabilities."""
    return np.asarray(
        [float((probs or {}).get(HATE_LABEL, 0.0)) for probs in probabilities], dtype=float
    )


def _binary_gold(y_true: Sequence[str]) -> np.ndarray:
    return np.asarray([1 if str(value) == HATE_LABEL else 0 for value in y_true], dtype=int)


def binary_average_precision(y_true: Sequence[str], scores: Sequence[float]) -> Optional[float]:
    """PR-AUC (average precision) for hate vs rest. None when one class is absent.

    PR-AUC is the right summary for this target: with ~5.8% positives, ROC-AUC is
    flattering and accuracy is dominated by the majority class.
    """
    gold = _binary_gold(y_true)
    if gold.sum() == 0 or gold.sum() == len(gold):
        return None
    return float(average_precision_score(gold, np.asarray(scores, dtype=float)))


def pr_curve_points(y_true: Sequence[str], scores: Sequence[float]) -> Dict[str, List[float]]:
    """Precision/recall coordinates for the hate-vs-rest PR curve."""
    gold = _binary_gold(y_true)
    precision, recall, _ = precision_recall_curve(gold, np.asarray(scores, dtype=float))
    return {"precision": [float(v) for v in precision], "recall": [float(v) for v in recall]}


def binary_decision_metrics(
    y_true: Sequence[str], decisions: Sequence[str], positive: str = HATE_LABEL
) -> Dict[str, Any]:
    """Everything for a model that answers hate-vs-rest directly (2-option choice, noul).

    Such a model never emits 'offensive language' or 'neither', so it has no
    three-class confusion matrix and must not be scored as if it did.
    """
    gold = _binary_gold(y_true)
    pred = np.asarray([1 if str(value) == positive else 0 for value in decisions], dtype=int)
    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    n = int(gold.size)
    return {
        "n": n,
        "accuracy": float((tp + tn) / n) if n else 0.0,
        "per_class": {},
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "hate": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(gold.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        },
        "labels": [positive, "not hate speech"],
    }


def operating_point(
    y_true: Sequence[str],
    probabilities: Sequence[Optional[Dict[str, float]]],
    labels: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """The threshold on P(hate) that maximises hate F1, and the 3-class metrics there.

    Laya (and any model that returns a distribution) is not obliged to decide by
    argmax. This quantifies how much the operating point matters for a target where
    hate speech is ~6% of the rows.
    """
    label_list = list(labels or CANONICAL_LABELS)
    if not probabilities:
        return None
    gold = _binary_gold(y_true)
    if gold.sum() == 0 or gold.sum() == len(gold):
        return None
    scores = hate_score_array(probabilities)
    precision, recall, thresholds = precision_recall_curve(gold, scores)
    if len(thresholds) == 0:
        return None
    f1 = np.where(
        (precision + recall) > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0
    )
    best = int(np.argmax(f1[:-1]))
    threshold = float(thresholds[best])

    others = [label for label in label_list if label != HATE_LABEL]
    predicted: List[str] = []
    for probs in probabilities:
        probs = probs or {}
        if float(probs.get(HATE_LABEL, 0.0)) >= threshold:
            predicted.append(HATE_LABEL)
        elif others:
            predicted.append(max(others, key=lambda label: float(probs.get(label, 0.0))))
        else:
            predicted.append(HATE_LABEL)

    metrics = compute_metrics(y_true, predicted, label_list)
    return {
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "hate_precision": metrics["hate"]["precision"],
        "hate_recall": metrics["hate"]["recall"],
        "hate_f1": metrics["hate"]["f1"],
        "hate_tp": metrics["hate"]["tp"],
        "hate_fp": metrics["hate"]["fp"],
        "hate_fn": metrics["hate"]["fn"],
    }


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #


def _core_metrics_fast(y_true_codes: np.ndarray, y_pred_codes: np.ndarray) -> tuple[float, float, float]:
    """(accuracy, macro F1, hate F1) straight from a confusion matrix.

    A vectorised twin of :func:`compute_metrics` for the bootstrap loop, where the
    sklearn call overhead would dominate a thousand resamples.
    """
    k = len(CANONICAL_LABELS)
    n = len(y_true_codes)
    if n == 0:
        return 0.0, 0.0, 0.0
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (y_true_codes, y_pred_codes), 1)
    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1e-12), 0.0)
    return float(tp.sum() / n), float(f1.mean()), float(f1[_HATE_CODE])


def _core_binary_fast(gold: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """(accuracy, hate F1) for a hate-vs-rest problem."""
    n = len(gold)
    if n == 0:
        return 0.0, 0.0
    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return (tp + tn) / n, f1


def bootstrap_intervals(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    n_samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
    task: str = "three_class",
) -> Dict[str, Any]:
    """Percentile bootstrap CIs for accuracy, macro F1, hate F1 and PR-AUC.

    ``task`` selects the estimand: a binary model has no macro F1, so a three-class
    macro F1 is not reported for it. One split is one sample; this says how much these
    numbers would move on a different sample of the same size. It is not a significance
    test between models (the splits are shared, so the comparison is paired).
    """
    true_arr = np.asarray([str(v) for v in y_true])
    pred_arr = np.asarray([str(v) for v in y_pred])
    n = len(true_arr)
    if n == 0:
        return {"samples": 0}
    score_array = np.asarray(scores, dtype=float) if scores is not None else None

    binary = task == TASK_HATE_BINARY
    if binary:
        gold_bin = (true_arr == HATE_LABEL).astype(int)
        pred_bin = (pred_arr == HATE_LABEL).astype(int)
    else:
        true_codes = np.asarray([_LABEL_TO_CODE.get(v, -1) for v in true_arr], dtype=int)
        pred_codes = np.asarray([_LABEL_TO_CODE.get(v, -1) for v in pred_arr], dtype=int)
        valid = (true_codes >= 0) & (pred_codes >= 0)
        true_codes, pred_codes = true_codes[valid], pred_codes[valid]
        if score_array is not None:
            score_array = score_array[valid]
        gold_bin = (true_codes == _HATE_CODE).astype(int)
        n = len(true_codes)

    rng = np.random.default_rng(seed)
    accuracies: List[float] = []
    macro_f1s: List[float] = []
    hate_f1s: List[float] = []
    average_precisions: List[float] = []

    for _ in range(n_samples):
        draw = rng.integers(0, n, n)
        if binary:
            accuracy, hate_f1 = _core_binary_fast(gold_bin[draw], pred_bin[draw])
        else:
            accuracy, macro_f1, hate_f1 = _core_metrics_fast(true_codes[draw], pred_codes[draw])
            macro_f1s.append(macro_f1)
        accuracies.append(accuracy)
        hate_f1s.append(hate_f1)
        if score_array is not None:
            sample_gold = gold_bin[draw]
            if 0 < int(sample_gold.sum()) < len(sample_gold):
                average_precisions.append(float(average_precision_score(sample_gold, score_array[draw])))

    def interval(values: List[float]) -> Optional[Dict[str, Any]]:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return None
        low = float(np.percentile(array, 100.0 * (1.0 - confidence) / 2.0))
        high = float(np.percentile(array, 100.0 * (1.0 + confidence) / 2.0))
        return {"mean": float(array.mean()), "low": low, "high": high, "n": int(array.size)}

    return {
        "samples": int(n_samples),
        "seed": int(seed),
        "confidence": float(confidence),
        "accuracy": interval(accuracies),
        "macro_f1": interval(macro_f1s) if not binary else None,
        "hate_f1": interval(hate_f1s),
        "average_precision": interval(average_precisions) if score_array is not None else None,
    }


# --------------------------------------------------------------------------- #
# Latency and reporting helpers
# --------------------------------------------------------------------------- #


def latency_stats(latencies_ms: Sequence[float]) -> Dict[str, Any]:
    """Total, mean, p50, p95 and throughput for a list of per-item timings."""
    values = np.asarray(list(latencies_ms), dtype=float)
    if values.size == 0:
        return {}
    total_s = float(values.sum() / 1000.0)
    # Below ~10 ms of total measured time the Python loop and the timer dominate,
    # so an "items/s" figure would be measuring the harness rather than the model.
    throughput = float(values.size / total_s) if total_s >= 0.01 else None
    return {
        "count": int(values.size),
        "total_s": round(total_s, 3),
        "average_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "throughput_per_s": throughput,
    }


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def format_float(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def comparison_table(metrics_by_model: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rows behind the comparison table (PRD section 21 plus the v1 additions).

    Binary models have no three-class macro F1 or per-class rows; those fields are
    None here rather than invented.
    """
    rows: List[Dict[str, Any]] = []
    for key, entry in metrics_by_model.items():
        latency = entry.get("latency") or {}
        bootstrap = entry.get("bootstrap") or {}
        macro_ci = bootstrap.get("macro_f1") or {}
        operating = entry.get("operating_point") or {}
        hate = entry.get("hate") or {}
        rows.append(
            {
                "model": key,
                "task": entry.get("task", "three_class"),
                "accuracy": entry.get("accuracy"),
                "macro_f1": entry.get("macro_f1"),
                "macro_precision": entry.get("macro_precision"),
                "macro_recall": entry.get("macro_recall"),
                "weighted_f1": entry.get("weighted_f1"),
                "hate_precision": hate.get("precision"),
                "hate_recall": hate.get("recall"),
                "hate_f1": hate.get("f1"),
                "hate_average_precision": hate.get("average_precision"),
                "hate_support": hate.get("support"),
                "hate_fp": hate.get("fp"),
                "hate_fn": hate.get("fn"),
                "macro_f1_ci_low": macro_ci.get("low"),
                "macro_f1_ci_high": macro_ci.get("high"),
                "operating_threshold": operating.get("threshold"),
                "operating_hate_f1": operating.get("hate_f1"),
                "operating_accuracy": operating.get("accuracy"),
                "load_time_s": entry.get("load_time_s"),
                "p50_latency_ms": latency.get("p50_ms"),
                "p95_latency_ms": latency.get("p95_ms"),
                "average_latency_ms": latency.get("average_ms"),
                "throughput_per_s": latency.get("throughput_per_s"),
            }
        )
    return rows
