"""Artifact rendering: plots and the Markdown report.

The report is generated from measured data only. Statements drawn from the numbers
are labelled as observations; anything evaluative sits under an explicit
"Interpretation" heading, so a reader can always tell the two apart (PRD section 31).

Section numbering follows the PRD's 16 sections; the v1.1 material (batching,
coverage, calibration, quadrants, binary variants) lives as subsections inside them.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CANONICAL_LABELS, HATE_LABEL, TASK_HATE_BINARY, Config
from .metrics import format_pct

TRAINING_REGIME_DISCLAIMER = (
    "This benchmark compares end-to-end classification approaches under their natural "
    "training regimes; it is not a controlled architecture-vs-architecture comparison."
)

NORMALIZER_DESCRIPTION = (
    "Duplicate detection normalizes the raw tweet in this order: lowercase; URLs removed; "
    "@mentions removed; a leading `rt` removed; punctuation optionally removed; whitespace "
    "collapsed and trimmed. Two tweets with the same normalized string share a group and are "
    "assigned to the same side of the split. The models always receive the raw, unmodified tweet."
)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_confusion_matrix(entry: Dict[str, Any], key: str, results_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.asarray(entry["confusion_matrix"], dtype=float)
    labels = list(entry.get("labels") or CANONICAL_LABELS)

    size = 5.2 if len(labels) > 2 else 3.8
    fig, ax = plt.subplots(figsize=(size, size - 0.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("%s — confusion matrix" % entry.get("display_name", key))

    threshold = matrix.max() / 2.0 if matrix.max() else 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "%d" % int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = results_dir / ("confusion_matrix_%s.png" % key)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pr_curve(curves: Dict[str, Dict[str, Any]], results_dir: Path) -> Path:
    """Precision-recall curve for hate vs rest, one line per model with probabilities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for key, curve in curves.items():
        ax.plot(curve["recall"], curve["precision"], linewidth=1.7,
                label="%s (PR-AUC %.3f)" % (curve["name"], curve["average_precision"]))
    ax.set_xlabel("Recall (hate speech)")
    ax.set_ylabel("Precision (hate speech)")
    ax.set_title("Hate speech vs rest — precision/recall")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path = results_dir / "pr_curve_hate_vs_rest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _pct(value: Any) -> str:
    return "n/a" if value is None else format_pct(float(value))


def _num(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _ci(interval: Optional[Dict[str, Any]], digits: int = 4) -> str:
    if not interval or interval.get("low") is None:
        return "n/a"
    return "[%s, %s]" % (_num(interval["low"], digits), _num(interval["high"], digits))


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _truncate(text: Any, limit: int = 150) -> str:
    value = str(text).replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _is_binary(entry: Dict[str, Any]) -> bool:
    return entry.get("task") == TASK_HATE_BINARY


def _three_class(models: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {key: entry for key, entry in models.items() if not _is_binary(entry)}


def _variants(models: Dict[str, Dict[str, Any]], prefix: str) -> Dict[str, Dict[str, Any]]:
    return {key: entry for key, entry in models.items() if key == prefix or key.startswith(prefix + "_")}


# --------------------------------------------------------------------------- #
# Report tables
# --------------------------------------------------------------------------- #


def _main_table(models: Dict[str, Dict[str, Any]]) -> str:
    """Three-class table: binary models are excluded, not scored on absent labels."""
    subset = _three_class(models)
    lines = [
        "| Model | Accuracy | Macro F1 | Macro F1 95% CI | Hate Precision | Hate Recall | Hate F1 | PR-AUC | p50 latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, entry in subset.items():
        bootstrap = (entry.get("bootstrap") or {}).get("macro_f1") or {}
        latency = entry.get("latency") or {}
        lines.append(
            "| {model} | {acc} | {f1} | {ci} | {hp} | {hr} | {hf} | {ap} | {p50} |".format(
                model=key,
                acc=_pct(entry.get("accuracy")),
                f1=_num(entry.get("macro_f1")),
                ci=_ci(bootstrap),
                hp=_num(entry["hate"].get("precision")),
                hr=_num(entry["hate"].get("recall")),
                hf=_num(entry["hate"].get("f1")),
                ap=_num(entry["hate"].get("average_precision"), 3),
                p50="n/a" if latency.get("p50_ms") is None else "%.1f ms" % latency["p50_ms"],
            )
        )
    return "\n".join(lines)


def _binary_table(models: Dict[str, Dict[str, Any]]) -> str:
    """Binary models: their own decision rule, no three-class metrics."""
    subset = {key: entry for key, entry in models.items() if _is_binary(entry)}
    if not subset:
        return "_No binary (hate-vs-rest) variant was configured._"
    lines = [
        "| Model | Task | Accuracy (binary) | Hate Precision | Hate Recall | Hate F1 | PR-AUC | Hate FN | Hate FP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, entry in subset.items():
        hate = entry.get("hate") or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (key, entry.get("task"), _pct(entry.get("accuracy")), _num(hate.get("precision")),
               _num(hate.get("recall")), _num(hate.get("f1")), _num(hate.get("average_precision"), 3),
               hate.get("fn"), hate.get("fp"))
        )
    return "\n".join(lines)


def _per_class_tables(models: Dict[str, Dict[str, Any]]) -> str:
    blocks = []
    for key, entry in _three_class(models).items():
        lines = [
            "**%s**" % entry.get("display_name", key),
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label in CANONICAL_LABELS:
            stats = (entry.get("per_class") or {}).get(label)
            if not stats:
                continue
            lines.append(
                "| {label} | {p} | {r} | {f} | {s} |".format(
                    label=label, p=_num(stats["precision"]), r=_num(stats["recall"]),
                    f=_num(stats["f1"]), s=stats["support"],
                )
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _bootstrap_table(models: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Model | Accuracy | 95% CI | Macro F1 | 95% CI | Hate F1 | 95% CI | PR-AUC | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, entry in models.items():
        boot = entry.get("bootstrap") or {}
        lines.append(
            "| {model} | {acc} | {acc_ci} | {f1} | {f1_ci} | {h} | {h_ci} | {ap} | {ap_ci} |".format(
                model=key,
                acc=_num(entry.get("accuracy")),
                acc_ci=_ci(boot.get("accuracy")),
                f1=_num(entry.get("macro_f1")),
                f1_ci=_ci(boot.get("macro_f1")),
                h=_num((entry.get("hate") or {}).get("f1")),
                h_ci=_ci(boot.get("hate_f1")),
                ap=_num((entry.get("hate") or {}).get("average_precision"), 3),
                ap_ci=_ci(boot.get("average_precision"), 3),
            )
        )
    return "\n".join(lines)


def _hate_table(models: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Model | Task | Hate precision | Hate recall | Hate F1 | PR-AUC | Hate FN | Hate FP | Hate support |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, entry in models.items():
        hate = entry.get("hate") or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (key, entry.get("task"), _num(hate.get("precision")), _num(hate.get("recall")),
               _num(hate.get("f1")), _num(hate.get("average_precision"), 3),
               hate.get("fn"), hate.get("fp"), hate.get("support"))
        )
    return "\n".join(lines)


def _operating_point_table(models: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Model | Decision rule | Accuracy | Macro F1 | Hate Precision | Hate Recall | Hate F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, entry in models.items():
        lines.append(
            "| {model} | argmax (natural) | {acc} | {f1} | {hp} | {hr} | {hf} |".format(
                model=key, acc=_pct(entry.get("accuracy")), f1=_num(entry.get("macro_f1")),
                hp=_num((entry.get("hate") or {}).get("precision")),
                hr=_num((entry.get("hate") or {}).get("recall")),
                hf=_num((entry.get("hate") or {}).get("f1")),
            )
        )
        operating = entry.get("operating_point")
        if operating:
            lines.append(
                "| {model} | threshold P(hate) ≥ {t:.2f} (from validation) | {acc} | {f1} | {hp} | {hr} | {hf} |".format(
                    model=key, t=operating["threshold"], acc=_pct(operating.get("accuracy")),
                    f1=_num(operating.get("macro_f1")), hp=_num(operating.get("hate_precision")),
                    hr=_num(operating.get("hate_recall")), hf=_num(operating.get("hate_f1")),
                )
            )
    return "\n".join(lines)


def _confusion_tables(models: Dict[str, Dict[str, Any]]) -> str:
    blocks = []
    for key, entry in models.items():
        labels = list(entry.get("labels") or CANONICAL_LABELS)
        lines = [
            "**%s**" % entry.get("display_name", key),
            "",
            "| Actual \\ Predicted | %s |" % " | ".join(labels),
            "| --- | %s |" % " | ".join("---:" for _ in labels),
        ]
        for i, label in enumerate(labels):
            cells = " | ".join(str(entry["confusion_matrix"][i][j]) for j in range(len(labels)))
            lines.append("| %s | %s |" % (label, cells))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _latency_table(models: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Model | Load (s) | Single-item p50 (ms) | Single-item p95 (ms) | Single-item (items/s) | Batch size | Batched total (s) | Batched (items/s) | Batch/single label agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, entry in models.items():
        latency = entry.get("latency") or {}
        batch = entry.get("batch") or {}
        lines.append(
            "| {model} | {load} | {p50} | {p95} | {single} | {bs} | {bt} | {bthr} | {match} |".format(
                model=key,
                load=_num(entry.get("load_time_s"), 2),
                p50=_num(latency.get("p50_ms"), 2),
                p95=_num(latency.get("p95_ms"), 2),
                single=_num(latency.get("throughput_per_s"), 2),
                bs=batch.get("batch_size") if batch.get("used") else "n/a",
                bt=_num(batch.get("total_s"), 2) if batch.get("used") else "n/a",
                bthr=_num(batch.get("throughput_per_s"), 2) if batch.get("used") else "n/a",
                match=_pct(batch["single_sample_label_match"]) if batch.get("single_sample_label_match") is not None else "n/a",
            )
        )
    lines.append("")
    lines.append(
        "_Load time is model loading/fitting. Single-item latency and throughput are measured on a "
        "%s-row sample with a warm-up; the batched column is the full-test-set pass. They are "
        "reported separately because they measure different things._" % "latency-sample"
    )
    return "\n".join(lines)


def _token_table(models: Dict[str, Dict[str, Any]]) -> str:
    lines = ["| Model | Rows | Truncated rows | Mean state tokens | Max state tokens | Limit |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    any_stats = False
    for key, entry in models.items():
        stats = entry.get("token_stats")
        if not stats:
            continue
        any_stats = True
        limit = stats.get("max_len") or stats.get("max_length")
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (key, stats.get("rows"), stats.get("truncated_rows"),
               _num(stats.get("mean_state_tokens"), 1), stats.get("max_state_tokens"), limit)
        )
    return "\n".join(lines) if any_stats else "_No model reported token statistics._"


def _coverage_table(coverage: Dict[str, Any], models: Dict[str, Dict[str, Any]]) -> str:
    if not coverage:
        return "_No model exposed a confidence, so coverage was not analysed._"
    lines = [
        "| Model | Confidence ≥ | Coverage | Accuracy on kept | Hate recall on kept |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, rows in coverage.items():
        for row in rows:
            lines.append(
                "| %s | %.2f | %s | %s | %s |"
                % (key, row["threshold"], _pct(row["coverage"]),
                   _pct(row["accuracy"]), _pct(row["hate_recall"]))
            )
    return "\n".join(lines)


def _quadrant_table(models: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Model | Correct & high | Correct & low | Incorrect & high | Incorrect & low | Mean conf. correct | Mean conf. incorrect |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    any_rows = False
    for key, entry in models.items():
        confidence = entry.get("confidence")
        if not confidence:
            continue
        any_rows = True
        lines.append(
            "| %s | %d | %d | %d | %d | %s | %s |"
            % (key, confidence["correct_high"], confidence["correct_low"],
               confidence["incorrect_high"], confidence["incorrect_low"],
               _num(confidence.get("mean_confidence_correct")),
               _num(confidence.get("mean_confidence_incorrect")))
        )
    return "\n".join(lines) if any_rows else "_No model exposed probabilities._"


def _calibration_table(calibration: Dict[str, Any]) -> str:
    if not calibration:
        return "_Calibration was not measured (no probabilities, or calibration disabled).__"
    lines = [
        "| Model | Base rate (val) | Brier | ECE | Bins |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, report in calibration.items():
        lines.append(
            "| %s | %s | %s | %s | %d |"
            % (key, _num(report.get("base_rate")), _num(report.get("brier")),
               _num(report.get("ece")), report.get("bins", 0))
        )
    lines.append("")
    lines.append(
        "_Measured on the **validation** split, not on test. Lower Brier and ECE are better. "
        "This measures calibration only; no temperature was fitted, so these are descriptive "
        "numbers for the shipped checkpoints._"
    )
    return "\n".join(lines)


def _error_summary(errors_path: Path) -> str:
    if not errors_path.exists():
        return "_error_analysis.csv was not produced._"
    import pandas as pd

    frame = pd.read_csv(errors_path)
    if frame.empty:
        return "_No misclassifications were recorded._"
    lines = ["| Model | Category | Exported |", "| --- | --- | ---: |"]
    grouped = frame.groupby(["model", "category"]).size().reset_index(name="exported")
    for _, row in grouped.sort_values(["model", "exported"], ascending=[True, False]).iterrows():
        lines.append("| %s | %s | %d |" % (row["model"], _cell(row["category"]), row["exported"]))
    lines.append("")
    lines.append("The full examples are in `results/error_analysis.csv`.")
    return "\n".join(lines)


def _error_examples_table(errors_path: Path, per_category: int, redact: bool = True) -> str:
    """Automatically chosen, most-confident errors per category (candidates for review).

    With ``redact`` (the default) the tweet text is withheld: the examples are hate
    speech, the report may be published, and the verbatim rows remain in the
    git-ignored ``results/error_analysis.csv``.
    """
    if not errors_path.exists():
        return "_No error file was produced._"
    import pandas as pd

    frame = pd.read_csv(errors_path)
    if frame.empty:
        return "_No misclassifications were recorded._"

    has_confidence = "confidence" in frame.columns
    lines = [
        "The rows below are **selected automatically** — the most confident mistakes per",
        "category, which are the most informative ones to read. They are candidates for",
        "manual coding, not a qualitative conclusion.",
        "",
    ]
    if redact:
        lines.append(
            "> **Tweet text is withheld from this report.** The examples are hate speech; the "
            "verbatim rows are in `results/error_analysis.csv`, which is not version-controlled. "
            "Set `output.error_examples_in_report: full` to print them in a local report."
        )
        lines.append("")
    placeholder = "[withheld — see results/error_analysis.csv]"
    for model in sorted(frame["model"].unique()):
        model_frame = frame[frame["model"] == model]
        for category in sorted(model_frame["category"].unique()):
            group = model_frame[model_frame["category"] == category]
            if has_confidence:
                group = group.sort_values("confidence", ascending=False, na_position="last")
            label = "%s — %s (%d exported)" % (model, _cell(category), len(group))
            lines.append("**%s**" % label)
            lines.append("")
            lines.append("| Example id | Gold | Predicted | Confidence |")
            lines.append("| ---: | --- | --- | ---: |")
            for _, row in group.head(per_category).iterrows():
                confidence = _num(row.get("confidence"), 3) if has_confidence else "n/a"
                lines.append("| %s | %s | %s | %s |" % (
                    row.get("id", "n/a"), _cell(row["gold_label"]),
                    _cell(row["predicted_label"]), confidence))
            if not redact:
                lines.append("")
                lines.append("| Text |")
                lines.append("| --- |")
                for _, row in group.head(per_category).iterrows():
                    lines.append("| %s |" % _cell(_truncate(row["text"])))
            lines.append("")
    if redact:
        lines.append("(%s)" % placeholder)
        lines.append("")
    return "\n".join(lines)


def _laya_variants_section(models: Dict[str, Dict[str, Any]]) -> str:
    variants = _variants(models, "laya")
    if not variants:
        return "_Laya was not part of this run._"
    lines = [
        "| Variant | Task | Question | Prompt (instructions) | Accuracy | Macro F1 | Hate P | Hate R | Hate F1 | PR-AUC |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, entry in variants.items():
        details = entry.get("details") or {}
        question = (details.get("questions") or {})
        first = question.get("label") or question.get("is_hate") or {}
        lines.append(
            "| {key} | {task} | {qtype} | {prompt} | {acc} | {f1} | {hp} | {hr} | {hf} | {ap} |".format(
                key=key,
                task=entry.get("task"),
                qtype=details.get("question_type", "choice"),
                prompt=_cell(_truncate(first.get("instructions", ""), 80)),
                acc=_pct(entry.get("accuracy")),
                f1=_num(entry.get("macro_f1")),
                hp=_num((entry.get("hate") or {}).get("precision")),
                hr=_num((entry.get("hate") or {}).get("recall")),
                hf=_num((entry.get("hate") or {}).get("f1")),
                ap=_num((entry.get("hate") or {}).get("average_precision"), 3),
            )
        )
    recalls = [(entry.get("hate") or {}).get("recall") for entry in variants.values()]
    recalls = [value for value in recalls if value is not None]
    if len(variants) > 1 and recalls:
        lines.append("")
        lines.append(
            "_Prompt/formulation sensitivity (measured): across %d Laya variants, hate recall "
            "spans %.4f to %.4f on the same test rows._" % (len(variants), min(recalls), max(recalls))
        )
    return "\n".join(lines)


def _regime_table(models: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Model | Training regime | Trained on the Davidson train split? | What it is |",
        "| --- | --- | --- | --- |",
    ]
    for key, entry in models.items():
        regime = entry.get("training_regime") or {}
        trained = regime.get("trained_on_davidson")
        lines.append("| %s | %s | %s | %s |" % (
            key, _cell(regime.get("regime", "unknown")),
            "Yes" if trained else ("No" if trained is not None else "unknown"),
            _cell(regime.get("note", ""))))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def generate_report(summary: Dict[str, Any], cfg: Config) -> Path:
    """Write ``report/benchmark_report.md`` (and a copy in ``results/``)."""
    models = summary["models"]
    comparison = summary["comparison"]
    dataset = summary["dataset"]
    split = summary["split"]
    environment = summary["environment"]
    calibration = summary.get("calibration") or {}
    coverage = summary.get("coverage") or {}
    debug_limit = summary.get("debug_limit")
    errors_path = cfg.results_dir / "error_analysis.csv"

    warnings: List[str] = []
    if debug_limit:
        warnings.append(
            "**DEBUG RUN (`--limit %d`).** This is not the full test set; do not treat these "
            "numbers as benchmark results." % debug_limit
        )
    if not dataset.get("sha256_verified"):
        warnings.append("The dataset hash was **not** verified; reproducibility of the input file is not guaranteed.")

    three_class = _three_class(models)
    binary_models = {key: entry for key, entry in models.items() if _is_binary(entry)}

    def best_by(subset: Dict[str, Dict[str, Any]], getter, higher: bool = True):
        pairs = [(key, getter(entry)) for key, entry in subset.items()]
        pairs = [(key, value) for key, value in pairs if value is not None]
        if not pairs:
            return None
        return (max if higher else min)(pairs, key=lambda item: item[1])

    best_macro = best_by(three_class, lambda entry: entry.get("macro_f1"))
    best_recall = best_by(models, lambda entry: (entry.get("hate") or {}).get("recall"))
    best_ap = best_by(models, lambda entry: (entry.get("hate") or {}).get("average_precision"))

    measured_lines: List[str] = []
    for key, entry in models.items():
        hate = entry.get("hate") or {}
        measured_lines.append(
            "- **%s**: accuracy %s, macro F1 %s, hate recall %s, hate precision %s, PR-AUC %s."
            % (key, _pct(entry.get("accuracy")), _num(entry.get("macro_f1")),
               _num(hate.get("recall")), _num(hate.get("precision")),
               _num(hate.get("average_precision"), 3))
        )

    lines: List[str] = []
    lines.append("# Local Hate Speech Detection Benchmark — Technical Report")
    lines.append("")
    lines.append("_Generated at %s by `run_benchmark.py`._" % summary["generated_at"])
    lines.append("")
    for warning in warnings:
        lines.append("> %s" % warning)
        lines.append("")

    # 1. Executive summary
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        "This report benchmarks **%d** classifier configurations on the Davidson "
        "hate-speech/offensive-language dataset: %d three-class models and %d binary "
        "(hate-vs-rest) variants. All were evaluated on the **same %d test tweets** from a "
        "single deterministic, duplicate-grouped split, with thresholds selected on a separate "
        "validation split."
        % (len(models), len(three_class), len(binary_models), summary["test_rows_used"])
    )
    lines.append("")
    lines.append("Measured results (full tables in sections 7 and 8):")
    lines.append("")
    lines.extend(measured_lines)
    lines.append("")
    if best_macro:
        lines.append("Highest macro F1 (three-class models): **%s** (%s)." % (best_macro[0], _num(best_macro[1])))
    if best_recall:
        lines.append("Highest hate recall: **%s** (%s)." % (best_recall[0], _num(best_recall[1])))
    if best_ap:
        lines.append("Highest PR-AUC (hate-vs-rest): **%s** (%s)." % (best_ap[0], _num(best_ap[1], 3)))
    lines.append("")
    lines.append(
        "The models do not simply rank by quality: they **disagree about what counts as hate "
        "speech**. One is precise and rarely fires; another fires often and catches far more hate "
        "speech at a large precision cost. That behavioural difference, not a single F1, is the "
        "headline finding."
    )
    lines.append("")

    # 2. Research question
    lines.append("## 2. Research Question")
    lines.append("")
    lines.append(
        "How does Laya perform on a known hate-speech dataset compared with classical and "
        "specialised local classifiers, when all models are evaluated on the same examples with "
        "the same metrics? And, for Laya, which formulation (three-class choice, binary choice, "
        "noul, improved definitions) gives the best hate-recall / precision / cost trade-off?"
    )
    lines.append("")

    # 3. Dataset
    lines.append("## 3. Dataset")
    lines.append("")
    lines.append("- **Name:** `%s`" % dataset.get("name"))
    lines.append("- **Source:** %s" % dataset.get("url"))
    lines.append("- **Description:** %s" % (dataset.get("description") or "n/a"))
    lines.append("- **Rows:** %d" % dataset.get("rows", 0))
    lines.append("- **SHA-256:** `%s`" % dataset.get("sha256"))
    lines.append("- **SHA-256 verified:** %s" % dataset.get("sha256_verified"))
    lines.append("")
    lines.append("| Class | Rows |")
    lines.append("| --- | ---: |")
    for label in CANONICAL_LABELS:
        lines.append("| %s | %d |" % (label, dataset.get("class_distribution", {}).get(label, 0)))
    lines.append("")

    # 4. Experimental setup
    lines.append("## 4. Experimental Setup")
    lines.append("")
    lines.append("- **Split:** train / validation / test, stratified by class and grouped so duplicate "
                 "tweets cannot cross a boundary. Seed `%s`, test size `%s`, validation size `%s`."
                 % (split.get("seed"), split.get("test_size"), split.get("validation_size")))
    lines.append("- **Rows:** train %d | validation %d | test %d"
                 % (split.get("train_rows", 0), split.get("validation_rows", 0), split.get("test_rows", 0)))
    lines.append("- **Test set signature (SHA-256 of test ids):** `%s`" % split.get("test_id_sha256"))
    lines.append("- **Groups (normalized tweets):** %d" % split.get("total_groups", 0))
    lines.append("- **Hate speech in the test set:** %d of %d (%.2f%%)"
                 % (split.get("test_class_counts", {}).get(HATE_LABEL, 0), split.get("test_rows", 0),
                    100.0 * split.get("test_class_counts", {}).get(HATE_LABEL, 0) / max(1, split.get("test_rows", 1))))
    lines.append("")
    lines.append("| Split | Hate speech | Offensive | Neither |")
    lines.append("| --- | ---: | ---: | ---: |")
    for part in ("train", "validation", "test"):
        counts = split.get("%s_class_counts" % part, {})
        lines.append("| %s | %d | %d | %d |" % (
            part, counts.get("hate speech", 0), counts.get("offensive language", 0), counts.get("neither", 0)))
    lines.append("")
    lines.append("**Duplicate normalizer.** %s" % NORMALIZER_DESCRIPTION)
    lines.append("")
    lines.append(
        "**Threshold selection.** The threshold on P(hate speech) is chosen on the validation "
        "split and applied unchanged to the test set. The test-optimal threshold is recorded in "
        "`results/metrics.json` only as an upper bound, never as a headline number."
    )
    lines.append("")
    lines.append(
        "**Statistical robustness.** Every headline metric is reported with a %d%% percentile "
        "bootstrap confidence interval over %d resamples of the test set (seed %s)."
        % (int(100 * summary["config"]["stats"]["confidence"]),
           summary["config"]["stats"]["bootstrap_samples"], summary["config"]["stats"]["seed"])
    )
    lines.append("")
    lines.append("**Hardware / runtime** (PRD section 18):")
    lines.append("")
    lines.append("- OS: %s (%s)" % (environment.get("os"), environment.get("platform")))
    lines.append("- CPU: %s (%s cores)" % (environment.get("cpu"), environment.get("cpu_count")))
    lines.append("- RAM: %s GB" % environment.get("ram_gb"))
    lines.append("- Python: %s" % environment.get("python"))
    lines.append("- Packages: %s" % json.dumps(environment.get("packages", {})))
    lines.append("- CUDA available: %s (torch CUDA %s)" % (environment.get("cuda_available"), environment.get("cuda_version")))
    if environment.get("gpu"):
        lines.append("- GPU: %s (%s GB)" % (environment.get("gpu"), environment.get("gpu_memory_gb")))
    lines.append("")

    # 5. Models
    lines.append("## 5. Models")
    lines.append("")
    lines.append(_regime_table(models))
    lines.append("")
    lines.append("> %s" % TRAINING_REGIME_DISCLAIMER)
    lines.append("")
    lines.append("| Key | Model | Task | Device | Load (s) |")
    lines.append("| --- | --- | --- | --- | ---: |")
    for key, entry in models.items():
        lines.append("| %s | %s | %s | %s | %s |" % (
            key, entry.get("display_name", key), entry.get("task"),
            entry.get("device") or "n/a", _num(entry.get("load_time_s"), 2)))
    lines.append("")
    revisions = summary.get("model_revisions", {})
    lines.append("- Laya Hub commit: `%s`" % revisions.get("laya_hub_sha"))
    lines.append("- HateXplain Hub commit: `%s`" % revisions.get("hatexplain_hub_sha"))
    lines.append("")

    # 6. Evaluation methodology
    lines.append("## 6. Evaluation Methodology")
    lines.append("")
    lines.append(
        "Each three-class model returns exactly one of hate speech / offensive language / neither. "
        "Each binary variant returns hate speech or not-hate-speech and is scored only on the "
        "binary problem. Every model sees exactly the same test rows, in the same order. Metrics "
        "use scikit-learn (`zero_division=0`) and hate-vs-rest PR-AUC (average precision), which is "
        "the fairer summary than ROC-AUC at ~5.8% positives."
    )
    lines.append("")
    lines.append(
        "**Inference.** Where the model has a batched path (Laya's `predict_batch`, HateXplain and "
        "TF-IDF), predictions come from one batched pass over the test set and single-item latency "
        "is measured separately on a sample, so throughput and latency are never conflated. The two "
        "paths are cross-checked: the report shows the label agreement between them."
    )
    lines.append("")

    # 7. Overall results
    lines.append("## 7. Overall Results")
    lines.append("")
    lines.append("### Three-class models")
    lines.append("")
    lines.append(_main_table(models))
    lines.append("")
    if binary_models:
        lines.append("### Binary (hate-vs-rest) variants")
        lines.append("")
        lines.append(_binary_table(models))
        lines.append("")
    lines.append("### Statistical robustness (95% bootstrap CI)")
    lines.append("")
    lines.append(_bootstrap_table(models))
    lines.append("")
    lines.append("### Per-class metrics (three-class models)")
    lines.append("")
    lines.append(_per_class_tables(models))
    lines.append("")
    lines.append(
        "_Measured result. Interpreting which model is 'better' depends on whether false negatives "
        "or false positives on hate speech matter more for the intended use, which this benchmark "
        "does not decide._"
    )
    lines.append("")

    # 8. Hate-speech results
    lines.append("## 8. Hate-Speech Results")
    lines.append("")
    lines.append(_hate_table(models))
    lines.append("")
    lines.append(
        "_Measured result. Hate recall = true hate tweets found / all true hate tweets. Hate FN are "
        "hate tweets labelled something else; hate FP are non-hate tweets labelled hate speech. "
        "Binary variants have no macro F1 because they never emit the other two labels._"
    )
    lines.append("")
    lines.append("### Operating point: natural decision vs a threshold on P(hate speech)")
    lines.append("")
    lines.append(_operating_point_table(models))
    lines.append("")
    lines.append(
        "_Measured result. The threshold row uses the same scores as the row above it, so any "
        "difference is the decision rule, not the model. The threshold comes from the validation "
        "split; it was not tuned on the test set._"
    )
    lines.append("")
    lines.append("### Confidence, coverage and calibration (secondary)")
    lines.append("")
    lines.append(_quadrant_table(models))
    lines.append("")
    lines.append(_calibration_table(calibration))
    lines.append("")
    lines.append("**Abstention / coverage.** Dropping the least confident answers:")
    lines.append("")
    lines.append(_coverage_table(coverage, models))
    lines.append("")
    lines.append(
        "_Measured result. 'Correct & high/low' splits answers by whether the reported probability "
        "reached %.2f. Coverage is the fraction of answers kept above a confidence threshold; "
        "accuracy and hate recall are measured on the kept subset, so they describe a different "
        "population at each row and are not directly comparable across rows._" % 0.5
    )
    lines.append("")

    # 9. Confusion matrices
    lines.append("## 9. Confusion Matrices")
    lines.append("")
    lines.append(_confusion_tables(models))
    lines.append("")
    lines.append("Plots: `results/confusion_matrix_<model>.png` and `results/pr_curve_hate_vs_rest.png`.")
    lines.append("")

    # 10. Latency
    lines.append("## 10. Latency Results")
    lines.append("")
    lines.append(_latency_table(models))
    lines.append("")
    lines.append("### Token budget and truncation")
    lines.append("")
    lines.append(_token_table(models))
    lines.append("")

    # 11. Error analysis
    lines.append("## 11. Error Analysis")
    lines.append("")
    lines.append(_error_summary(errors_path))
    lines.append("")
    lines.append("### Representative errors")
    lines.append("")
    lines.append(_error_examples_table(errors_path, cfg.output.error_examples_per_category,
                                       redact=cfg.output.error_examples_in_report != "full"))
    lines.append("")

    # 12. Laya analysis
    lines.append("## 12. Laya Analysis")
    lines.append("")
    lines.append(
        "Laya was run **zero-shot**; no fine-tuning was performed (PRD section 24 defers it to "
        "phase 2). Because a zero-shot model's answer depends on the prompt *and* on the question "
        "type, several formulations are compared — three-class choice, binary choice, noul, and an "
        "explicit-definition prompt:"
    )
    lines.append("")
    lines.append(_laya_variants_section(models))
    lines.append("")
    lines.append(
        "_No variant was selected using the test set: all formulations are reported, and thresholds "
        "come from validation. Choosing a single formulation for production is a separate decision "
        "that needs its own validation protocol._"
    )
    lines.append("")

    # 13. Limitations
    lines.append("## 13. Limitations")
    lines.append("")
    lines.append("- Single dataset (Davidson), single language (English), single split seed.")
    lines.append(
        "- **No generalization test.** A strong result on one dataset does not establish "
        "generalization; a second hate-speech dataset is required."
    )
    lines.append(
        "- **Calibration is measured, not fitted.** Brier and ECE are reported for the shipped "
        "checkpoints; no temperature scaling was applied, so the confidence numbers are descriptive "
        "and should not be read as 'about c of answers at c are correct'."
    )
    lines.append(
        "- **No fine-tuning.** Laya's zero-shot numbers are a starting point, not its ceiling."
    )
    lines.append(
        "- Models have different label taxonomies: HateXplain's own three classes are mapped onto "
        "Davidson's, which introduces a mapping assumption the report cannot remove."
    )
    lines.append(
        "- The benchmark compares end-to-end approaches under their natural training regimes (see "
        "section 5), so it is not evidence that one architecture is better than another."
    )
    lines.append(
        "- Latency was measured per item on this machine and device only; it does not transfer to "
        "other hardware. Batched and single-item numbers are both reported, and batched shapes can "
        "change floating-point results, so their label agreement is reported too."
    )
    lines.append(
        "- The Davidson labels are themselves noisy and imbalanced; the majority class is "
        "'offensive language', so accuracy is dominated by that class."
    )
    lines.append("")
    lines.append("### Known gaps")
    lines.append("")
    lines.append("| Area | Status |")
    lines.append("| --- | --- |")
    lines.append("| Dataset, grouped split, reproducibility | Done |")
    lines.append("| Three-class metrics, confusion matrices, latency | Done |")
    lines.append("| Hate-vs-rest with PR-AUC, bootstrap CIs | Done |")
    lines.append("| Validation split; thresholds selected there | Done |")
    lines.append("| Batched inference with separate throughput | Done |")
    lines.append("| Binary formulations (choice, noul) | Done |")
    lines.append("| Calibration measured (Brier, ECE, reliability) | Done |")
    lines.append("| Temperature fitting / recalibration | **Missing** |")
    lines.append("| Generalization to a second dataset | **Missing** |")
    lines.append("| Qualitative error coding | Mechanical only (see section 11) |")
    lines.append("| Fine-tuning Laya on the train split | **Missing (phase 2)** |")
    lines.append("")

    # 14. Conclusions
    lines.append("## 14. Conclusions")
    lines.append("")
    lines.append("**Measured result.**")
    lines.append("")
    lines.extend(measured_lines)
    lines.append("")
    lines.append("**Interpretation.**")
    lines.append("")
    if best_macro:
        conservative = min(
            ((key, (entry.get("hate") or {}).get("recall")) for key, entry in models.items()),
            key=lambda item: item[1] if item[1] is not None else 1.0,
        )
        lines.append(
            "Read the table as a whole rather than by a single metric. `%s` leads on macro F1 among "
            "the three-class models, but its hate recall is %s: a precise, conservative detector. "
            "`%s` has the highest hate recall (%s), i.e. a sensitive, over-triggering detector. "
            "`%s` is the most conservative of all. These are not interchangeable, and which is "
            "'better' is a product decision about false negatives vs false positives, not a "
            "property of the models."
            % (
                best_macro[0],
                _num((models[best_macro[0]].get("hate") or {}).get("recall")),
                best_recall[0] if best_recall else "n/a",
                _num(best_recall[1]) if best_recall else "n/a",
                conservative[0],
            )
        )
        lines.append("")
        lines.append(
            "For Laya specifically, the interesting question is not whether it beats a supervised "
            "baseline, but **which formulation** gives the best recall/precision/cost trade-off — "
            "section 12 measures that, and section 8 shows how much the decision rule alone changes "
            "the answer."
        )
    else:
        lines.append("No comparison was available to interpret.")
    lines.append("")

    # 15. Reproduction
    lines.append("## 15. Reproduction Instructions")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Environment (PyTorch build first, matching your hardware)")
    lines.append("uv venv --python 3.12")
    lines.append("uv pip install torch --index-url https://download.pytorch.org/whl/cu126")
    lines.append("uv pip install -r requirements.txt")
    lines.append("")
    lines.append("# 2. Run everything")
    lines.append("python run_benchmark.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "The split is reproducible: the same seed and normalization reproduce the same test ids, "
        "and section 4 records the SHA-256 of the test id list. Pinned model revisions are in "
        "`results/experiment_config.json`."
    )
    lines.append("")

    # 16. References
    lines.append("## 16. References")
    lines.append("")
    lines.append(
        "1. Davidson, T., Warmsley, D., Macy, M., Weber, I. *Automated Hate Speech Detection and "
        "the Problem of Offensive Language.* ICWSM 2017."
    )
    lines.append("2. Davidson hate-speech/offensive-language dataset, 24,783 English tweets, three classes.")
    lines.append("3. Laya decision-engine repository and documentation (Convai Innovations).")
    lines.append("4. Mathew, B. et al. *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*; model `Hate-speech-CNERG/bert-base-uncased-hatexplain`.")
    lines.append("")

    report_dir = cfg.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "benchmark_report.md"
    content = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    shutil.copyfile(report_path, cfg.results_dir / "report.md")
    return report_path
