"""Experiment orchestration: dataset -> split -> models -> metrics -> artifacts.

``run_benchmark`` is the single function that turns a :class:`Config` into the
artifact set described in PRD section 20. It is deterministic given the config
(seed, split, model revisions) and writes everything under ``results/`` and
``report/``.
"""
from __future__ import annotations

import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .config import CANONICAL_LABELS, Config, PROJECT_ROOT, PROBABILITY_SLUG, config_to_dict
from .dataset import build_split, load_raw, save_processed, write_json
from .environment import runtime_info
from .metrics import comparison_table, compute_metrics, latency_stats
from .models import ModelResult, build_models
from . import reporting

Logger = Callable[[str], None]


def _log(message: str) -> None:  # pragma: no cover - default logger
    print(message, flush=True)


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
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


def _run_model(
    key: str,
    classifier,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    warmup: int,
    log: Logger,
) -> ModelResult:
    """Load, warm up, then classify every test tweet one at a time, timing each."""
    log("model [%s]: loading / fitting ..." % key)
    started = time.perf_counter()
    classifier.load(train_df)
    load_time = time.perf_counter() - started
    log("model [%s]: ready in %.2fs (%s)" % (key, load_time, classifier.display_name))

    texts = test_df["text"].tolist()
    for text in texts[: max(0, warmup)]:
        classifier.predict_one(text)

    wants_probabilities = classifier.supports_probabilities()
    labels: List[str] = []
    probabilities: Optional[List[Optional[Dict[str, float]]]] = [] if wants_probabilities else None
    latencies: List[float] = []
    for index, text in enumerate(texts):
        started = time.perf_counter()
        label, probs = classifier.predict_one(text)
        latencies.append((time.perf_counter() - started) * 1000.0)
        labels.append(label)
        if probabilities is not None:
            probabilities.append(probs)
        if (index + 1) % 500 == 0:
            log("model [%s]: %d/%d predicted" % (key, index + 1, len(texts)))

    device = getattr(classifier, "device", None)
    revision = getattr(classifier, "revision", None)
    return ModelResult(
        key=key,
        name=classifier.display_name,
        labels=labels,
        probabilities=probabilities,
        latencies_ms=latencies,
        load_time_s=load_time,
        device=str(device) if device is not None else None,
        revision=str(revision) if revision is not None else None,
        details=classifier.details(),
    )


def _assemble_predictions(
    test_df: pd.DataFrame, results: Dict[str, ModelResult]
) -> pd.DataFrame:
    frame = test_df[["id", "text", "gold_label"]].copy()
    for key, result in results.items():
        frame[f"{key}_prediction"] = result.labels
        if result.probabilities is not None:
            for label in CANONICAL_LABELS:
                slug = PROBABILITY_SLUG[label]
                frame[f"{key}_{slug}_probability"] = [
                    (probs or {}).get(label) for probs in result.probabilities
                ]
            frame[f"{key}_confidence"] = [
                max(probs.values()) if probs else None for probs in result.probabilities
            ]
        frame[f"{key}_latency_ms"] = result.latencies_ms
    return frame


def _error_analysis(
    predictions: pd.DataFrame, results: Dict[str, ModelResult], per_category: int
) -> pd.DataFrame:
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
        # Deterministic order: by size, then alphabetically.
        for (gold, predicted), positions in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0])
        ):
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
    test_df = frame[frame["split"] == "test"].reset_index(drop=True)
    if limit is not None and limit > 0:
        # Dev-only: a deterministic head of the test set for smoke runs.
        test_df = test_df.head(limit).reset_index(drop=True)
        log("split: DEBUG limit applied -> %d test rows" % len(test_df))

    write_json(results_dir / "dataset_metadata.json", dataset_metadata)
    split_artifact = dict(split_info)
    write_json(results_dir / "split.json", split_artifact)

    models = build_models(cfg, only)
    if not models:
        raise RuntimeError("no models enabled; check config.yaml and --models")
    log("models: %s" % ", ".join(key for key, _ in models))

    results: Dict[str, ModelResult] = {}
    for key, classifier in models:
        results[key] = _run_model(key, classifier, train_df, test_df, cfg.inference_warmup, log)

    predictions = _assemble_predictions(test_df, results)
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    log("artifacts: predictions.csv (%d rows)" % len(predictions))

    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    for key, result in results.items():
        entry = compute_metrics(predictions["gold_label"], predictions[f"{key}_prediction"])
        entry["latency"] = latency_stats(result.latencies_ms)
        entry["load_time_s"] = round(result.load_time_s, 3)
        entry["display_name"] = result.name
        entry["device"] = result.device
        entry["revision"] = result.revision
        entry["details"] = result.details

        # Secondary confidence analysis (PRD section 23): mean probability of the
        # reported answer, correct vs incorrect. Not a calibration curve.
        confidence_column = f"{key}_confidence"
        if confidence_column in predictions.columns:
            confidence = predictions[confidence_column]
            if confidence.notna().any():
                correct = predictions["gold_label"] == predictions[f"{key}_prediction"]
                entry["confidence"] = {
                    "correct_mean": float(confidence[correct].mean()),
                    "incorrect_mean": float(confidence[~correct].mean()),
                    "overall_mean": float(confidence.mean()),
                }
        metrics_by_model[key] = entry

    comparison_rows = comparison_table(metrics_by_model)
    pd.DataFrame(comparison_rows).to_csv(results_dir / "model_comparison.csv", index=False)
    write_json(results_dir / "metrics.json", {"models": metrics_by_model, "comparison": comparison_rows})
    log("artifacts: metrics.json, model_comparison.csv")

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
        "model_revisions": {
            "laya_hub_sha": laya_sha,
            "hatexplain_hub_sha": hatexplain_sha,
        },
        "debug_limit": limit,
        "test_rows_used": int(len(test_df)),
        "test_id_sha256": split_info["test_id_sha256"],
    }
    write_json(results_dir / "experiment_config.json", experiment_config)
    log("artifacts: experiment_config.json")

    summary: Dict[str, Any] = {
        "generated_at": started_at,
        "config": config_to_dict(cfg),
        "environment": environment,
        "dataset": dataset_metadata,
        "split": {k: v for k, v in split_info.items() if k not in ("train_ids", "test_ids")},
        "models": metrics_by_model,
        "comparison": comparison_rows,
        "debug_limit": limit,
        "test_rows_used": int(len(test_df)),
        "model_revisions": experiment_config["model_revisions"],
        "project_git_commit": experiment_config["project_git_commit"],
        "results_dir": str(results_dir),
    }

    report_path = reporting.generate_report(summary, cfg)
    log("artifacts: %s" % report_path)
    return summary
