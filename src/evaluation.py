"""Experiment orchestration: dataset -> split -> models -> metrics -> artifacts.

``run_benchmark`` turns a :class:`Config` into the artifact set described in PRD
section 20 plus the v1.1 additions:

* a train/validation/test split (validation carved out of train, test unchanged),
* batched prediction with a *separate* single-item latency measurement,
* binary hate-vs-rest variants (2-option choice, noul),
* thresholds selected on validation and applied to test,
* coverage/abstention, confidence quadrants, token/truncation stats,
* calibration (Brier, ECE, reliability) measured on validation.
"""
from __future__ import annotations

import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import reporting
from .calibration import calibration_report
from .config import (
    CANONICAL_LABELS,
    PROBABILITY_SLUG,
    PROJECT_ROOT,
    TASK_HATE_BINARY,
    TASK_THREE_CLASS,
    TRAINING_REGIME,
    Config,
    config_to_dict,
)
from .dataset import build_split, load_raw, save_processed, write_json
from .environment import runtime_info
from .metrics import (
    binary_average_precision,
    binary_decision_metrics,
    bootstrap_intervals,
    comparison_table,
    compute_metrics,
    hate_score_array,
    latency_stats,
    operating_point,
    pr_curve_points,
)
from .models import ModelResult, build_models

Logger = Callable[[str], None]
PredictionRow = Tuple[str, Optional[Dict[str, float]]]

# Confidence at or above which an answer counts as "high confidence" in the
# quadrant analysis.
HIGH_CONFIDENCE = 0.5


def _log(message: str) -> None:  # pragma: no cover - default logger
    print(message, flush=True)


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _hf_sha(repo_id: str, revision: Optional[str] = None) -> Optional[str]:
    """The resolved Hub commit for a model repo, for exact reproducibility."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id, revision=revision or "main").sha
    except Exception:
        return None


def _regime_for(key: str) -> Dict[str, Any]:
    """How this model key relates to the Davidson training split."""
    if key in TRAINING_REGIME:
        return dict(TRAINING_REGIME[key])
    if key.startswith("laya"):
        return dict(TRAINING_REGIME["laya"])
    if key.startswith("tfidf"):
        return dict(TRAINING_REGIME["tfidf"])
    return {"regime": "unknown", "trained_on_davidson": None, "note": ""}


def _plain_predict(classifier, texts: List[str]) -> Tuple[List[str], Optional[List[Optional[Dict[str, float]]]]]:
    """Predictions without timing, using the batched path when the model has one."""
    batched = classifier.predict_batch(texts)
    if batched is not None:
        return [row[0] for row in batched], ([row[1] for row in batched] if classifier.supports_probabilities() else None)
    labels: List[str] = []
    probabilities = [] if classifier.supports_probabilities() else None
    for text in texts:
        label, probs = classifier.predict_one(text)
        labels.append(label)
        if probabilities is not None:
            probabilities.append(probs)
    return labels, probabilities


def _token_stats(classifier) -> Optional[Dict[str, Any]]:
    stats_fn = getattr(classifier, "token_stats", None)
    return stats_fn() if callable(stats_fn) else None


def _run_model(
    key: str,
    classifier,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    log: Logger,
) -> Tuple[ModelResult, Dict[str, Any]]:
    """Load, predict, and measure. Returns the test result plus the validation scores."""
    log("model [%s]: loading / fitting ..." % key)
    started = time.perf_counter()
    classifier.load(train_df)
    load_time = time.perf_counter() - started
    log("model [%s]: ready in %.2fs (%s)" % (key, load_time, classifier.display_name))

    test_texts = test_df["text"].tolist()
    validation_texts = validation_df["text"].tolist()

    # Validation pass: used only to select the threshold and to measure calibration.
    validation_labels, validation_probs = _plain_predict(classifier, validation_texts)
    reset = getattr(classifier, "reset_stats", None)
    if callable(reset):
        reset()

    batch_info: Optional[Dict[str, Any]] = None
    mode = cfg.inference.predict_mode
    labels: List[str] = []
    probabilities: Optional[List[Optional[Dict[str, float]]]] = None
    latencies: List[float] = []
    token_stats: Optional[Dict[str, Any]] = None

    if mode == "batch":
        batch_started = time.perf_counter()
        batched = classifier.predict_batch(test_texts)
        batch_seconds = time.perf_counter() - batch_started
        if batched is not None:
            labels = [row[0] for row in batched]
            probabilities = [row[1] for row in batched] if classifier.supports_probabilities() else None
            batch_info = {
                "used": True,
                "n": len(batched),
                "batch_size": getattr(getattr(classifier, "cfg", None), "batch_size", None),
                "total_s": batch_seconds,
                "throughput_per_s": (len(batched) / batch_seconds) if batch_seconds else None,
            }
            # Capture token/truncation stats for the test pass only, before the
            # latency sample adds more single-item calls.
            token_stats = _token_stats(classifier)
            log("model [%s]: batched %d states in %.2fs" % (key, len(batched), batch_seconds))
        else:
            mode = "single"

    if mode == "single":
        for text in test_texts[: max(0, cfg.inference.warmup)]:
            classifier.predict_one(text)
        probabilities = [] if classifier.supports_probabilities() else None
        for text in test_texts:
            started = time.perf_counter()
            label, probs = classifier.predict_one(text)
            latencies.append((time.perf_counter() - started) * 1000.0)
            labels.append(label)
            if probabilities is not None:
                probabilities.append(probs)
        log("model [%s]: single-item pass over %d states" % (key, len(test_texts)))
        token_stats = _token_stats(classifier)
    else:
        # Batched predictions are authoritative; per-item latency is measured on a
        # sample so throughput and latency are never conflated.
        sample = test_texts[: max(0, cfg.inference.latency_sample)]
        for text in sample[: max(0, cfg.inference.warmup)]:
            classifier.predict_one(text)
        sample_labels: List[str] = []
        for text in sample:
            started = time.perf_counter()
            label, _ = classifier.predict_one(text)
            latencies.append((time.perf_counter() - started) * 1000.0)
            sample_labels.append(label)
        if batch_info is not None and sample_labels:
            matches = sum(1 for a, b in zip(sample_labels, labels[: len(sample_labels)]) if a == b)
            batch_info["single_sample_n"] = len(sample_labels)
            batch_info["single_sample_label_match"] = matches / len(sample_labels)
            log(
                "model [%s]: single-item sample %d, batched/single label agreement %.4f"
                % (key, len(sample_labels), batch_info["single_sample_label_match"])
            )

    result = ModelResult(
        key=key,
        name=classifier.display_name,
        labels=labels,
        task=getattr(classifier, "task", TASK_THREE_CLASS),
        probabilities=probabilities,
        latencies_ms=latencies,
        load_time_s=load_time,
        device=str(getattr(classifier, "device", None)) if getattr(classifier, "device", None) else None,
        revision=str(getattr(classifier, "revision", None)) if getattr(classifier, "revision", None) else None,
        batch=batch_info,
        token_stats=token_stats,
        details=classifier.details(),
    )
    unload = getattr(classifier, "unload", None)
    if callable(unload):
        unload()
    validation = {"labels": validation_labels, "probabilities": validation_probs}
    return result, validation


def _assemble_predictions(test_df: pd.DataFrame, results: Dict[str, ModelResult]) -> pd.DataFrame:
    frame = test_df[["id", "text", "gold_label"]].copy()
    for key, result in results.items():
        frame[f"{key}_prediction"] = result.labels
        if result.probabilities is not None:
            labels_seen: List[str] = []
            for probs in result.probabilities:
                for label in (probs or {}):
                    if label not in labels_seen:
                        labels_seen.append(label)
            for label in labels_seen:
                slug = PROBABILITY_SLUG.get(label, label.replace(" ", "_"))
                frame[f"{key}_{slug}_probability"] = [
                    (probs or {}).get(label) for probs in result.probabilities
                ]
            frame[f"{key}_confidence"] = [
                max(probs.values()) if probs else None for probs in result.probabilities
            ]
        frame[f"{key}_latency_ms"] = result.latencies_ms if len(result.latencies_ms) == len(frame) else None
    return frame


def _coverage_table(gold, predictions, confidences, thresholds) -> List[Dict[str, Any]]:
    """Coverage / accuracy / hate recall after dropping answers below a threshold."""
    import numpy as np

    rows: List[Dict[str, Any]] = []
    gold_array = np.asarray([str(v) for v in gold])
    pred_array = np.asarray([str(v) for v in predictions])
    confidence = np.asarray(confidences, dtype=float)
    for threshold in thresholds:
        keep = confidence >= threshold
        coverage = float(keep.mean())
        if keep.sum() == 0:
            rows.append({"threshold": threshold, "coverage": coverage, "n": 0,
                         "accuracy": None, "hate_recall": None, "hate_precision": None})
            continue
        kept_gold = gold_array[keep]
        kept_pred = pred_array[keep]
        accuracy = float((kept_gold == kept_pred).mean())
        true_hate = kept_gold == "hate speech"
        predicted_hate = kept_pred == "hate speech"
        tp = int((true_hate & predicted_hate).sum())
        fp = int((~true_hate & predicted_hate).sum())
        fn = int((true_hate & ~predicted_hate).sum())
        rows.append({
            "threshold": float(threshold),
            "coverage": coverage,
            "n": int(keep.sum()),
            "accuracy": accuracy,
            "hate_recall": (tp / (tp + fn)) if (tp + fn) else None,
            "hate_precision": (tp / (tp + fp)) if (tp + fp) else None,
        })
    return rows


def _confidence_quadrants(gold, predictions, confidences) -> Optional[Dict[str, Any]]:
    import numpy as np

    if confidences is None:
        return None
    gold_array = np.asarray([str(v) for v in gold])
    pred_array = np.asarray([str(v) for v in predictions])
    confidence = np.asarray(confidences, dtype=float)
    correct = gold_array == pred_array
    high = confidence >= HIGH_CONFIDENCE
    return {
        "high_confidence": HIGH_CONFIDENCE,
        "correct_high": int((correct & high).sum()),
        "correct_low": int((correct & ~high).sum()),
        "incorrect_high": int((~correct & high).sum()),
        "incorrect_low": int((~correct & ~high).sum()),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(confidence[~correct].mean()) if (~correct).any() else None,
    }


def _error_analysis(predictions: pd.DataFrame, results: Dict[str, ModelResult], per_category: int) -> pd.DataFrame:
    """Misclassified examples grouped by (gold -> predicted), capped per group."""
    rows: List[Dict[str, Any]] = []
    for key, result in results.items():
        column = f"{key}_prediction"
        confidence_column = f"{key}_confidence"
        subset = predictions[["id", "text", "gold_label", column]].copy()
        subset = subset[subset["gold_label"] != subset[column]]
        grouped: Dict[Tuple[str, str], List[int]] = {}
        for position, (_, row) in enumerate(subset.iterrows()):
            grouped.setdefault((str(row["gold_label"]), str(row[column])), []).append(position)
        for (gold, predicted), positions in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
            for position in positions[:per_category]:
                row = subset.iloc[position]
                entry: Dict[str, Any] = {
                    "model": key,
                    "category": f"{gold} -> {predicted}",
                    "gold_label": gold,
                    "predicted_label": predicted,
                    "text": row["text"],
                }
                if confidence_column in predictions.columns:
                    entry["confidence"] = predictions.iloc[row.name][confidence_column]
                if result.probabilities is not None:
                    probs = result.probabilities[row.name] or {}
                    entry["predicted_probability"] = probs.get(predicted)
                    entry["gold_probability"] = probs.get(gold)
                rows.append(entry)
    return pd.DataFrame(rows)


def run_benchmark(
    cfg: Config,
    only: Optional[List[str]] = None,
    limit: Optional[int] = None,
    log: Logger = _log,
) -> Dict[str, Any]:
    """Execute the whole experiment and return a summary dict."""
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    results_dir = cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    environment = runtime_info()
    log("environment: python %s, torch %s, laya %s, cuda=%s"
        % (environment["python"], environment["packages"]["torch"],
           environment["packages"]["laya"], environment.get("cuda_available")))

    raw_df, dataset_metadata = load_raw(cfg, log)
    frame, split_info = build_split(raw_df, cfg, log)
    save_processed(frame, log)

    train_df = frame[frame["split"] == "train"].reset_index(drop=True)
    validation_df = frame[frame["split"] == "validation"].reset_index(drop=True)
    test_df = frame[frame["split"] == "test"].reset_index(drop=True)
    if limit is not None and limit > 0:
        test_df = test_df.head(limit).reset_index(drop=True)
        validation_df = validation_df.head(limit).reset_index(drop=True)
        log("split: DEBUG limit applied -> %d test / %d validation rows" % (len(test_df), len(validation_df)))

    write_json(results_dir / "dataset_metadata.json", dataset_metadata)
    write_json(results_dir / "split.json", dict(split_info))

    models = build_models(cfg, only)
    if not models:
        raise RuntimeError("no models enabled; check config.yaml and --models")
    log("models: %s" % ", ".join(key for key, _ in models))

    results: Dict[str, ModelResult] = {}
    validation_by_model: Dict[str, Dict[str, Any]] = {}
    for key, classifier in models:
        results[key], validation_by_model[key] = _run_model(
            key, classifier, train_df, validation_df, test_df, cfg, log
        )

    predictions = _assemble_predictions(test_df, results)
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    log("artifacts: predictions.csv (%d rows)" % len(predictions))

    gold = predictions["gold_label"]
    validation_gold = validation_df["gold_label"]
    scores_by_model: Dict[str, Optional[Any]] = {
        key: (hate_score_array(result.probabilities) if result.probabilities is not None else None)
        for key, result in results.items()
    }

    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    calibration_by_model: Dict[str, Any] = {}
    coverage_by_model: Dict[str, Any] = {}

    for key, result in results.items():
        task = result.task
        model_predictions = predictions[f"{key}_prediction"]

        if task == TASK_HATE_BINARY:
            entry = binary_decision_metrics(gold, model_predictions)
        else:
            entry = compute_metrics(gold, model_predictions)
        entry["task"] = task
        entry["latency"] = latency_stats(result.latencies_ms)
        entry["load_time_s"] = round(result.load_time_s, 3)
        entry["display_name"] = result.name
        entry["device"] = result.device
        entry["revision"] = result.revision
        entry["details"] = result.details
        entry["training_regime"] = _regime_for(key)
        entry["batch"] = result.batch
        entry["token_stats"] = result.token_stats

        scores = scores_by_model[key]
        validation_scores = None
        if validation_by_model[key]["probabilities"] is not None:
            validation_scores = hate_score_array(validation_by_model[key]["probabilities"])

        if scores is not None:
            entry["hate"]["average_precision"] = binary_average_precision(gold, scores)
            # Threshold selected on validation, applied unchanged to test.
            if validation_scores is not None:
                selected = operating_point(validation_gold, validation_by_model[key]["probabilities"])
                if selected:
                    entry["operating_point"] = _apply_threshold(
                        gold, result.probabilities, selected["threshold"], task
                    )
                    entry["operating_point"]["threshold_source"] = "validation"
                    entry["operating_point"]["threshold_source_rows"] = int(len(validation_df))
                    # The test-optimal threshold is reported only as an upper bound,
                    # never as the headline number.
                    entry["operating_point_on_test"] = operating_point(gold, result.probabilities)
            if cfg.calibration.enabled and validation_scores is not None:
                calibration_by_model[key] = calibration_report(
                    validation_gold, validation_scores, bins=cfg.calibration.bins
                )

        confidence_column = f"{key}_confidence"
        if confidence_column in predictions.columns and predictions[confidence_column].notna().any():
            confidences = predictions[confidence_column].tolist()
            entry["confidence"] = _confidence_quadrants(gold, model_predictions, confidences)
            coverage = _coverage_table(gold, model_predictions, confidences,
                                       [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            coverage_by_model[key] = coverage

        entry["bootstrap"] = bootstrap_intervals(
            gold, model_predictions, scores=scores,
            n_samples=cfg.stats.bootstrap_samples, seed=cfg.stats.seed,
            confidence=cfg.stats.confidence, task=task,
        )
        metrics_by_model[key] = entry

    comparison_rows = comparison_table(metrics_by_model)
    pd.DataFrame(comparison_rows).to_csv(results_dir / "model_comparison.csv", index=False)
    write_json(results_dir / "metrics.json", {
        "models": metrics_by_model,
        "comparison": comparison_rows,
        "calibration": calibration_by_model,
        "coverage": coverage_by_model,
    })
    log("artifacts: metrics.json, model_comparison.csv")

    operating_rows = [
        dict({"model": key}, **(entry.get("operating_point") or {}))
        for key, entry in metrics_by_model.items()
        if entry.get("operating_point")
    ]
    if operating_rows:
        pd.DataFrame(operating_rows).to_csv(results_dir / "operating_points.csv", index=False)
        log("artifacts: operating_points.csv (%d models)" % len(operating_rows))

    if coverage_by_model:
        coverage_rows = []
        for key, rows in coverage_by_model.items():
            for row in rows:
                coverage_rows.append(dict({"model": key}, **row))
        pd.DataFrame(coverage_rows).to_csv(results_dir / "coverage.csv", index=False)
        log("artifacts: coverage.csv")

    curves: Dict[str, Dict[str, Any]] = {}
    for key, scores in scores_by_model.items():
        if scores is None:
            continue
        entry = metrics_by_model[key]
        average_precision = entry["hate"].get("average_precision")
        if average_precision is None:
            continue
        curves[key] = {
            "name": entry["display_name"],
            "average_precision": average_precision,
            **pr_curve_points(gold, scores),
        }
    if curves:
        reporting.plot_pr_curve(curves, results_dir)
        log("artifacts: pr_curve_hate_vs_rest.png")

    for key, entry in metrics_by_model.items():
        reporting.plot_confusion_matrix(entry, key, results_dir)
    log("artifacts: confusion matrices")

    errors = _error_analysis(predictions, results, cfg.output.error_samples_per_category)
    errors.to_csv(results_dir / "error_analysis.csv", index=False)
    log("artifacts: error_analysis.csv (%d rows)" % len(errors))

    laya_sha = _hf_sha(cfg.laya.repo, cfg.laya.revision) if cfg.laya.enabled else None
    hatexplain_sha = _hf_sha(cfg.hatexplain.model_id, cfg.hatexplain.revision) if cfg.hatexplain.enabled else None

    experiment_config = {
        "generated_at": started_at,
        "project_git_commit": _git_commit(),
        "config_path": cfg.path,
        "config": config_to_dict(cfg),
        "environment": environment,
        "model_revisions": {"laya_hub_sha": laya_sha, "hatexplain_hub_sha": hatexplain_sha},
        "debug_limit": limit,
        "test_rows_used": int(len(test_df)),
        "validation_rows_used": int(len(validation_df)),
        "test_id_sha256": split_info["test_id_sha256"],
    }
    write_json(results_dir / "experiment_config.json", experiment_config)
    log("artifacts: experiment_config.json")

    summary: Dict[str, Any] = {
        "generated_at": started_at,
        "config": config_to_dict(cfg),
        "environment": environment,
        "dataset": dataset_metadata,
        "split": {k: v for k, v in split_info.items() if k not in ("train_ids", "validation_ids", "test_ids")},
        "models": metrics_by_model,
        "comparison": comparison_rows,
        "calibration": calibration_by_model,
        "coverage": coverage_by_model,
        "training_regime": {key: entry["training_regime"] for key, entry in metrics_by_model.items()},
        "debug_limit": limit,
        "test_rows_used": int(len(test_df)),
        "validation_rows_used": int(len(validation_df)),
        "model_revisions": experiment_config["model_revisions"],
        "project_git_commit": experiment_config["project_git_commit"],
        "results_dir": str(results_dir),
    }
    # Persisted so the report can be re-rendered (e.g. after changing the redaction
    # setting) without re-running any model: see scripts/render_report.py.
    write_json(results_dir / "summary.json", summary)

    report_path = reporting.generate_report(summary, cfg)
    log("artifacts: %s" % report_path)
    return summary


def _apply_threshold(gold, probabilities, threshold: float, task: str) -> Dict[str, Any]:
    """3-class (or binary) metrics at a fixed threshold on P(hate speech)."""
    predicted: List[str] = []
    others = [label for label in CANONICAL_LABELS if label != "hate speech"]
    for probs in probabilities:
        probs = probs or {}
        if float(probs.get("hate speech", 0.0)) >= threshold:
            predicted.append("hate speech")
        elif task == TASK_HATE_BINARY:
            predicted.append("not hate speech")
        elif others:
            predicted.append(max(others, key=lambda label: float(probs.get(label, 0.0))))
        else:
            predicted.append("hate speech")

    if task == TASK_HATE_BINARY:
        metrics = binary_decision_metrics(gold, predicted)
        return {
            "threshold": float(threshold),
            "accuracy": metrics["accuracy"],
            "macro_f1": None,
            "hate_precision": metrics["hate"]["precision"],
            "hate_recall": metrics["hate"]["recall"],
            "hate_f1": metrics["hate"]["f1"],
            "hate_tp": metrics["hate"]["tp"],
            "hate_fp": metrics["hate"]["fp"],
            "hate_fn": metrics["hate"]["fn"],
        }
    metrics = compute_metrics(gold, predicted)
    return {
        "threshold": float(threshold),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "hate_precision": metrics["hate"]["precision"],
        "hate_recall": metrics["hate"]["recall"],
        "hate_f1": metrics["hate"]["f1"],
        "hate_tp": metrics["hate"]["tp"],
        "hate_fp": metrics["hate"]["fp"],
        "hate_fn": metrics["hate"]["fn"],
    }
