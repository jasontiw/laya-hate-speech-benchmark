"""Artifact rendering: confusion-matrix plots and the Markdown report.

The report is generated from ``metrics.json`` data only — every number comes from a
measured result. Statements drawn from those numbers are labelled as observations;
anything evaluative is put under an explicit "Interpretation" heading so a reader
can always tell the two apart (PRD section 31).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from .config import CANONICAL_LABELS, Config, HATE_LABEL
from .metrics import format_pct

PROBABILITY_SHORT = {"hate speech": "hate", "offensive language": "offensive", "neither": "neither"}


def plot_confusion_matrix(entry: Dict[str, Any], key: str, results_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.asarray(entry["confusion_matrix"], dtype=float)
    labels = list(entry["labels"])

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
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
            ax.text(
                j,
                i,
                "%d" % int(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = results_dir / ("confusion_matrix_%s.png" % key)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _pct(value: Any) -> str:
    return "n/a" if value is None else format_pct(float(value))


def _num(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _main_table(comparison: List[Dict[str, Any]]) -> str:
    header = (
        "| Model | Accuracy | Macro F1 | Hate Precision | Hate Recall | Hate F1 | Hate FN | Hate FP | p50 latency | p95 latency |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = []
    for row in comparison:
        rows.append(
            "| {model} | {acc} | {f1} | {hp} | {hr} | {hf} | {fn} | {fp} | {p50} | {p95} |".format(
                model=row["model"],
                acc=_pct(row["accuracy"]),
                f1=_num(row["macro_f1"]),
                hp=_num(row["hate_precision"]),
                hr=_num(row["hate_recall"]),
                hf=_num(row["hate_f1"]),
                fn=row["hate_fn"],
                fp=row["hate_fp"],
                p50="n/a" if row["p50_latency_ms"] is None else f"{row['p50_latency_ms']:.1f} ms",
                p95="n/a" if row["p95_latency_ms"] is None else f"{row['p95_latency_ms']:.1f} ms",
            )
        )
    return header + "\n" + "\n".join(rows)


def _per_class_tables(models: Dict[str, Dict[str, Any]]) -> str:
    blocks = []
    for key, entry in models.items():
        lines = [
            "**%s**" % entry.get("display_name", key),
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label in CANONICAL_LABELS:
            stats = entry["per_class"][label]
            lines.append(
                "| {label} | {p} | {r} | {f} | {s} |".format(
                    label=label,
                    p=_num(stats["precision"]),
                    r=_num(stats["recall"]),
                    f=_num(stats["f1"]),
                    s=stats["support"],
                )
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _confusion_tables(models: Dict[str, Dict[str, Any]]) -> str:
    blocks = []
    for key, entry in models.items():
        lines = [
            "**%s**" % entry.get("display_name", key),
            "",
            "| Actual \\ Predicted | %s |" % " | ".join(CANONICAL_LABELS),
            "| --- | %s |" % " | ".join("---:" for _ in CANONICAL_LABELS),
        ]
        for i, label in enumerate(CANONICAL_LABELS):
            cells = " | ".join(str(entry["confusion_matrix"][i][j]) for j in range(len(CANONICAL_LABELS)))
            lines.append("| %s | %s |" % (label, cells))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _latency_table(models: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Model | Load (s) | Total (s) | Mean (ms) | p50 (ms) | p95 (ms) | Throughput (items/s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, entry in models.items():
        latency = entry.get("latency") or {}
        lines.append(
            "| {model} | {load} | {total} | {mean} | {p50} | {p95} | {tp} |".format(
                model=key,
                load=_num(entry.get("load_time_s"), 2),
                total=_num(latency.get("total_s"), 2),
                mean=_num(latency.get("average_ms"), 2),
                p50=_num(latency.get("p50_ms"), 2),
                p95=_num(latency.get("p95_ms"), 2),
                tp=_num(latency.get("throughput_per_s"), 2),
            )
        )
    lines.append("")
    lines.append(
        "_Load time is model loading/fitting and is reported separately from inference "
        "(PRD section 17). Latency measures warm per-item inference only._"
    )
    return "\n".join(lines)


def _error_summary(errors_path: Path) -> str:
    if not errors_path.exists():
        return "_error_analysis.csv was not produced._"
    import pandas as pd

    frame = pd.read_csv(errors_path)
    if frame.empty:
        return "_No misclassifications were recorded._"
    lines = [
        "Counts of exported misclassification categories (capped per category):",
        "",
        "| Model | Category | Exported |",
        "| --- | --- | ---: |",
    ]
    grouped = frame.groupby(["model", "category"]).size().reset_index(name="count")
    grouped = grouped.sort_values(["model", "count"], ascending=[True, False])
    for _, row in grouped.iterrows():
        lines.append("| %s | %s | %d |" % (row["model"], row["category"], row["count"]))
    lines.append("")
    lines.append("The full examples are in `results/error_analysis.csv`.")
    return "\n".join(lines)


def _confidence_section(models: Dict[str, Dict[str, Any]]) -> str:
    rows = ["| Model | Mean confidence (correct) | Mean confidence (incorrect) | Mean confidence (all) |",
            "| --- | ---: | ---: | ---: |"]
    any_confidence = False
    for key, entry in models.items():
        confidence = entry.get("confidence")
        if not confidence:
            continue
        any_confidence = True
        rows.append(
            "| {model} | {c} | {i} | {a} |".format(
                model=key,
                c=_num(confidence.get("correct_mean")),
                i=_num(confidence.get("incorrect_mean")),
                a=_num(confidence.get("overall_mean")),
            )
        )
    if not any_confidence:
        return "_No model in this run exposed probabilities, so confidence was not analysed._"
    rows.append("")
    rows.append(
        "_Secondary analysis (PRD section 23). 'Confidence' here is the probability of the "
        "reported answer; it is not a calibration curve, and it is not comparable across "
        "models with different label sets or temperatures._"
    )
    return "\n".join(rows)


def generate_report(summary: Dict[str, Any], cfg: Config) -> Path:
    """Write ``report/benchmark_report.md`` (and a copy in ``results/``)."""
    models = summary["models"]
    comparison = summary["comparison"]
    dataset = summary["dataset"]
    split = summary["split"]
    environment = summary["environment"]
    debug_limit = summary.get("debug_limit")

    model_count = len(models)
    warnings: List[str] = []
    if debug_limit:
        warnings.append(
            "**DEBUG RUN (`--limit %d`).** This is not the full test set; do not treat these "
            "numbers as benchmark results." % debug_limit
        )
    if not dataset.get("sha256_verified"):
        warnings.append(
            "The dataset hash was **not** verified against an expected value; reproducibility "
            "of the input file is not guaranteed."
        )

    laya_present = "laya" in models
    laya = models.get("laya", {})
    best = max(comparison, key=lambda row: row["macro_f1"]) if comparison else None

    measured_lines: List[str] = []
    for row in comparison:
        measured_lines.append(
            "- **%s**: accuracy %s, macro F1 %s, hate recall %s, hate precision %s."
            % (
                row["model"],
                _pct(row["accuracy"]),
                _num(row["macro_f1"]),
                _num(row["hate_recall"]),
                _num(row["hate_precision"]),
            )
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
        "This report benchmarks **%d** classifiers on the Davidson hate-speech/offensive-language "
        "dataset. All comparable models were evaluated on the **same %d test tweets** produced by "
        "a single deterministic, duplicate-grouped split." % (model_count, summary["test_rows_used"])
    )
    lines.append("")
    lines.append("Measured results (see section 7 for the full table):")
    lines.append("")
    lines.extend(measured_lines)
    lines.append("")
    if best is not None:
        lines.append(
            "On this test set the highest macro F1 was **%s** (%s). The majority baseline is a "
            "reminder that accuracy alone is uninformative on this imbalanced dataset."
            % (best["model"], _num(best["macro_f1"]))
        )
    lines.append("")

    # 2. Research question
    lines.append("## 2. Research Question")
    lines.append("")
    lines.append(
        "How does Laya perform on a known hate-speech dataset compared with classical and "
        "specialised local classifiers, when all models are evaluated on the same examples with "
        "the same metrics?"
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
    lines.append("- **Loaded at:** %s" % dataset.get("loaded_at"))
    lines.append("")
    lines.append("| Class | Rows |")
    lines.append("| --- | ---: |")
    for label in CANONICAL_LABELS:
        lines.append("| %s | %d |" % (label, dataset.get("class_distribution", {}).get(label, 0)))
    lines.append("")

    # 4. Experimental setup
    lines.append("## 4. Experimental Setup")
    lines.append("")
    lines.append(
        "- **Split:** stratified by class, grouped so that duplicate tweets cannot cross the "
        "boundary. Seed `%s`, test size `%s`." % (split.get("seed"), split.get("test_size"))
    )
    lines.append("- **Train rows:** %d  |  **Test rows:** %d" % (split.get("train_rows", 0), split.get("test_rows", 0)))
    lines.append("- **Test set signature (SHA-256 of test ids):** `%s`" % split.get("test_id_sha256"))
    lines.append("- **Groups (normalized tweets):** %d" % split.get("total_groups", 0))
    lines.append("")
    lines.append("| Split | Hate speech | Offensive | Neither |")
    lines.append("| --- | ---: | ---: | ---: |")
    for part in ("train", "test"):
        counts = split.get("%s_class_counts" % part, {})
        lines.append(
            "| %s | %d | %d | %d |"
            % (part, counts.get("hate speech", 0), counts.get("offensive language", 0), counts.get("neither", 0))
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
    lines.append("| Key | Model | Device | Load (s) |")
    lines.append("| --- | --- | --- | ---: |")
    for key, entry in models.items():
        lines.append(
            "| %s | %s | %s | %s |"
            % (key, entry.get("display_name", key), entry.get("device") or "n/a", _num(entry.get("load_time_s"), 2))
        )
    lines.append("")
    revisions = summary.get("model_revisions", {})
    lines.append("- Laya Hub commit: `%s`" % revisions.get("laya_hub_sha"))
    lines.append("- HateXplain Hub commit: `%s`" % revisions.get("hatexplain_hub_sha"))
    if laya_present:
        lines.append("- Laya checkpoint requested: `%s` (resolved: `%s`)"
                     % (laya.get("details", {}).get("repo"), laya.get("details", {}).get("revision_resolved")))
    lines.append("")

    # 6. Evaluation methodology
    lines.append("## 6. Evaluation Methodology")
    lines.append("")
    lines.append(
        "Each model receives the raw tweet text and returns exactly one of the three canonical "
        "labels. Every model sees exactly the same test rows, in the same order. Metrics are "
        "computed with scikit-learn (`zero_division=0`). Hate speech is additionally treated as a "
        "binary problem (hate vs rest) to expose its precision, recall and error counts, which "
        "aggregate accuracy hides."
    )
    lines.append("")

    # 7. Overall results
    lines.append("## 7. Overall Results")
    lines.append("")
    lines.append(_main_table(comparison))
    lines.append("")
    lines.append("### Per-class metrics")
    lines.append("")
    lines.append(_per_class_tables(models))
    lines.append("")
    lines.append(
        "_Measured result. Interpreting which model is 'better' depends on whether false "
        "negatives or false positives on hate speech matter more for the intended use, which "
        "this benchmark does not decide._"
    )
    lines.append("")

    # 8. Hate-speech results
    lines.append("## 8. Hate-Speech Results")
    lines.append("")
    lines.append("| Model | Hate precision | Hate recall | Hate F1 | Hate FN | Hate FP | Hate support |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, entry in models.items():
        hate = entry["hate"]
        lines.append(
            "| %s | %s | %s | %s | %d | %d | %d |"
            % (key, _num(hate["precision"]), _num(hate["recall"]), _num(hate["f1"]), hate["fn"], hate["fp"], hate["support"])
        )
    lines.append("")
    lines.append(
        "_Measured result. Hate recall = true hate tweets found / all true hate tweets. "
        "Hate FN are hate tweets labelled something else; hate FP are non-hate tweets labelled "
        "hate speech._"
    )
    lines.append("")

    # 9. Confusion matrices
    lines.append("## 9. Confusion Matrices")
    lines.append("")
    lines.append(_confusion_tables(models))
    lines.append("")
    lines.append("Plots are written to `results/confusion_matrix_<model>.png`.")
    lines.append("")

    # 10. Latency
    lines.append("## 10. Latency Results")
    lines.append("")
    lines.append(_latency_table(models))
    lines.append("")

    # 11. Error analysis
    lines.append("## 11. Error Analysis")
    lines.append("")
    lines.append(_error_summary(cfg.results_dir / "error_analysis.csv"))
    lines.append("")

    # 12. Laya analysis
    lines.append("## 12. Laya Analysis")
    lines.append("")
    if laya_present:
        hate = laya["hate"]
        per_class = laya["per_class"]
        lines.append(
            "Laya was run **zero-shot** as a `choice` question; no fine-tuning was performed in "
            "this MVP (PRD section 24 defers it to phase 2). Measured on the shared test set:"
        )
        lines.append("")
        lines.append(
            "- Accuracy %s, macro F1 %s." % (_pct(laya["accuracy"]), _num(laya["macro_f1"]))
        )
        lines.append(
            "- Hate speech: precision %s, recall %s, F1 %s (%d false negatives, %d false positives)."
            % (_num(hate["precision"]), _num(hate["recall"]), _num(hate["f1"]), hate["fn"], hate["fp"])
        )
        lines.append(
            "- Per class recall: hate %s, offensive %s, neither %s."
            % (
                _num(per_class["hate speech"]["recall"]),
                _num(per_class["offensive language"]["recall"]),
                _num(per_class["neither"]["recall"]),
            )
        )
        lines.append(
            "- Inference: p50 %s ms, p95 %s ms, throughput %s items/s."
            % (
                _num(laya.get("latency", {}).get("p50_ms"), 2),
                _num(laya.get("latency", {}).get("p95_ms"), 2),
                _num(laya.get("latency", {}).get("throughput_per_s"), 2),
            )
        )
        lines.append("")
        lines.append("### Confidence (secondary)")
        lines.append("")
        lines.append(_confidence_section({"laya": laya}))
    else:
        lines.append("Laya was not part of this run.")
    lines.append("")

    # 13. Limitations
    lines.append("## 13. Limitations")
    lines.append("")
    lines.append("- Single dataset (Davidson), single language (English), single split seed.")
    lines.append(
        "- A strong result on one dataset does not establish generalisation; cross-dataset "
        "testing is a phase-2 item (PRD section 30)."
    )
    lines.append(
        "- Models have different label taxonomies: HateXplain's own three classes are mapped "
        "onto Davidson's, which introduces a mapping assumption the report cannot remove."
    )
    lines.append(
        "- Latency was measured per item on this machine and device only; it does not transfer "
        "to other hardware."
    )
    lines.append(
        "- The Davidson labels are themselves noisy and imbalanced; the majority class is "
        "'offensive language', so accuracy is dominated by that class."
    )
    lines.append(
        "- This is a research benchmark, not a moderation system; no threshold tuning, "
        "calibration fitting or deployment policy was applied."
    )
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
    if best is not None:
        lines.append(
            "Reading the table as a whole rather than by any single metric: `%s` has the highest "
            "macro F1 here, while hate-speech recall separates the models far more than accuracy "
            "does. The majority baseline shows why accuracy alone would be misleading, and "
            "Laya's zero-shot behaviour should be read as a starting point that fine-tuning "
            "(phase 2) is meant to change, not as its ceiling." % best["model"]
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
        "The split is reproducible: the same seed and normalization reproduce the same test "
        "ids, and section 4 records the SHA-256 of the test id list. Pinned model revisions are "
        "in `results/experiment_config.json`."
    )
    lines.append("")

    # 16. References
    lines.append("## 16. References")
    lines.append("")
    lines.append(
        "1. Davidson, T., Warmsley, D., Macy, M., Weber, I. *Automated Hate Speech Detection "
        "and the Problem of Offensive Language.* ICWSM 2017."
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
    # PRD section 20 lists report.md under results/ as well.
    shutil.copyfile(report_path, cfg.results_dir / "report.md")
    return report_path
