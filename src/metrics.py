"""Metrics: overall, per-class, hate-specific and latency (PRD sections 14, 15, 17)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .config import CANONICAL_LABELS, HATE_LABEL


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


def latency_stats(latencies_ms: Sequence[float]) -> Dict[str, Any]:
    """Total, mean, p50, p95 and throughput for a list of per-item timings."""
    values = np.asarray(list(latencies_ms), dtype=float)
    if values.size == 0:
        return {}
    total_s = float(values.sum() / 1000.0)
    return {
        "count": int(values.size),
        "total_s": round(total_s, 3),
        "average_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "throughput_per_s": float(values.size / total_s) if total_s else None,
    }


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def format_float(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def comparison_table(metrics_by_model: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rows behind the PRD section 21 table."""
    rows: List[Dict[str, Any]] = []
    for key, entry in metrics_by_model.items():
        latency = entry.get("latency") or {}
        rows.append(
            {
                "model": key,
                "accuracy": entry["accuracy"],
                "macro_f1": entry["macro_f1"],
                "macro_precision": entry["macro_precision"],
                "macro_recall": entry["macro_recall"],
                "weighted_f1": entry["weighted_f1"],
                "hate_precision": entry["hate"]["precision"],
                "hate_recall": entry["hate"]["recall"],
                "hate_f1": entry["hate"]["f1"],
                "hate_support": entry["hate"]["support"],
                "hate_fp": entry["hate"]["fp"],
                "hate_fn": entry["hate"]["fn"],
                "load_time_s": entry.get("load_time_s"),
                "p50_latency_ms": latency.get("p50_ms"),
                "p95_latency_ms": latency.get("p95_ms"),
                "average_latency_ms": latency.get("average_ms"),
                "throughput_per_s": latency.get("throughput_per_s"),
            }
        )
    return rows
