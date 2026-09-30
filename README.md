# Local Hate Speech Detection Benchmark

**English** | [Español](README.es.md)

A small, local, reproducible experiment that measures how **Laya** performs on
hate-speech detection and compares it against classical and specialised classifiers
on the **same public dataset** and the **same test rows**.

Research question:

> How does Laya perform on a known hate-speech dataset compared with other local
> classifiers on the same examples and metrics — and which **Laya formulation**
> (three-class choice, binary choice, `noul`, explicit definitions) gives the best
> hate-recall / precision / cost trade-off?

This is a research benchmark, **not** a production moderation system. It reduces to
one command and a set of generated artifacts.

---

## The finding this benchmark is built around

The models do not simply rank by quality — they **disagree about what counts as hate
speech**. On the same test rows, one model is precise and rarely fires; another fires
often and catches far more hate speech at a large precision cost. Aggregate accuracy
hides this, so the benchmark reports hate-vs-rest metrics, PR-AUC, an operating point
selected on validation, coverage, calibration and bootstrap intervals alongside the
usual table.

---

## Models compared

Eight configurations, from three conceptually different approaches:

| Key | Model | Task | Training regime | Trained on the Davidson train split? |
| --- | --- | --- | --- | --- |
| `majority` | Majority-class baseline | 3-class | Baseline | No |
| `tfidf` | TF-IDF + Logistic Regression | 3-class | Supervised (fit here) | Yes |
| `tfidf_balanced` | Same, `class_weight="balanced"` | 3-class | Supervised (fit here) | Yes |
| `laya` | Laya, 3-class choice — **L1** | 3-class | Zero-shot | No |
| `laya_semantic` | Laya, 3-class choice, explicit definitions — **L4** | 3-class | Zero-shot | No |
| `laya_binary` | Laya, 2-option choice hate/not-hate — **L2** | hate vs rest | Zero-shot | No |
| `laya_noul` | Laya, `noul` P(hate) — **L3** | hate vs rest | Zero-shot | No |
| `hatexplain` | `Hate-speech-CNERG/bert-base-uncased-hatexplain` | 3-class | Pretrained externally | No |

> **This benchmark compares end-to-end classification approaches under their natural
> training regimes; it is not a controlled architecture-vs-architecture comparison.**

In particular, `tfidf` is fit on the in-domain training split while `laya` and
`hatexplain` are used as published. A higher score for a supervised model is expected
and is not evidence that its architecture is better.

Dataset: **Davidson** (`tdavidson/hate_speech_offensive`), 24,783 English tweets,
three classes — `hate speech` (0), `offensive language` (1), `neither` (2).

---

## Quickstart

Requires Python 3.10+ and (for the CUDA build) an NVIDIA GPU.

```powershell
cd laya-hate-speech-benchmark

# 1. Environment — install the PyTorch build that matches your hardware FIRST
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu126   # or .../cpu
uv pip install -r requirements.txt

# 2. Reproduce everything
.\.venv\Scripts\python.exe run_benchmark.py
```

On macOS/Linux use `.venv/bin/python run_benchmark.py`.

The first run downloads the dataset (~2.5 MB) and the model checkpoints (Laya English,
HateXplain BERT); those are cached by Hugging Face afterwards.

### Useful flags

```bash
python run_benchmark.py --list-models               # print the configured model keys and exit
python run_benchmark.py --limit 200                 # dev smoke run (clearly marked as such)
python run_benchmark.py --models laya,laya_binary   # only some configurations
python run_benchmark.py --config my_experiment.yaml # alternative config
```

Every run is driven by [`config.yaml`](config.yaml); that file is copied verbatim into
`results/experiment_config.json`.

---

## Output artifacts

```
results/
├── dataset_metadata.json          # source, SHA-256, rows, class distribution
├── experiment_config.json         # config + hardware/runtime + resolved model revisions
├── split.json                     # seed, counts, and the exact train/validation/test ids
├── predictions.csv                # one row per test tweet, all predictions + probabilities
├── metrics.json                   # metrics, CI, operating points, calibration, coverage
├── model_comparison.csv           # the main comparison table
├── operating_points.csv           # natural decision vs validation-selected threshold
├── coverage.csv                   # coverage / accuracy / hate recall vs confidence
├── error_analysis.csv             # misclassified examples by (gold -> predicted)
├── confusion_matrix_<model>.png
├── pr_curve_hate_vs_rest.png      # hate-vs-rest precision/recall, one line per model
└── report.md

report/
└── benchmark_report.md            # the final deliverable
```

`benchmark_report.md` covers the PRD's 16 sections and, inside them, the v1.1 material:
training regimes, bootstrap CIs, binary variants, batching latency, coverage,
calibration and a known-gaps checklist.

---

## Methodology

- **Dataset handling.** Downloaded programmatically, SHA-256 verified, never edited,
  original labels preserved.
- **Split.** Stratified by class and grouped so duplicates cannot cross a boundary:
  **train / validation / test**. Validation is carved out of the training side, so the
  test set is unchanged and results stay comparable across runs. The normalizer, in order:

  ```python
  def normalize(text):
      value = text
      value = value.lower()                                  # lowercase
      value = re.sub(r"https?://\S+|www\.\S+", " ", value)   # URLs removed
      value = re.sub(r"@\w+", " ", value)                    # @mentions removed
      value = re.sub(r"^\s*rt\b[:\s]+", "", value, re.I)     # leading "rt" removed
      value = re.sub(r"[^\w\s]", " ", value)                 # punctuation removed (optional; off by default)
      value = re.sub(r"\s+", " ", value).strip()             # whitespace collapsed and trimmed
      return value
  ```

- **Laya formulations.** Three-class choice, binary choice, `noul` and an
  explicit-definition prompt, all zero-shot. None is selected using the test set.
- **HateXplain mapping.** Its `id2label` is read from the model config and mapped onto
  the canonical labels (`hate speech -> hate speech`, `offensive -> offensive language`,
  `normal -> neither`). The raw mapping is recorded in the report.
- **Metrics.** Accuracy, macro F1, per-class precision/recall/F1 and confusion matrices
  for three-class models; hate-vs-rest precision/recall/F1 with FP/FN counts for all.
- **Ranking metric.** **PR-AUC (average precision)** for hate vs rest. With ~5.8%
  positives, ROC-AUC is flattering and accuracy is dominated by the majority class.
- **Operating point.** The threshold on `P(hate speech)` is **selected on validation**
  and applied unchanged to test; the test-optimal threshold is recorded only as an
  upper bound.
- **Inference.** Where a batched path exists (Laya's `predict_batch`, HateXplain,
  TF-IDF), predictions come from one batched pass and **single-item latency is measured
  separately on a sample**, so throughput and latency are never conflated. The two paths
  are cross-checked and the label agreement is reported (batched shapes can change
  floating-point results).
- **Uncertainty.** Percentile **bootstrap 95% CIs** for accuracy, macro F1, hate F1 and
  PR-AUC. It describes variability, not significance between models (the splits are
  shared, so the comparison is paired).
- **Calibration & coverage.** Brier, ECE and reliability bins for the hate score,
  measured **on validation**; plus a coverage table showing accuracy and hate recall
  after dropping the least confident answers.
- **Token budget.** Mean/max state tokens and the number of truncated rows per model.

---

## Project layout

```
laya-hate-speech-benchmark/
├── config.yaml               # experiment settings (single source of truth)
├── run_benchmark.py          # entry point: python run_benchmark.py
├── requirements.txt
├── pyproject.toml
├── data/{raw,processed}/     # git-ignored
├── docs/PRD.md               # the specification this implements
├── scripts/
│   ├── run_benchmark.py      # thin wrapper (same entry point)
│   └── render_report.py      # re-render the report without re-running models
├── src/
│   ├── config.py             # typed config, canonical labels, tasks, training regimes
│   ├── dataset.py            # download, verify, normalize, grouped 3-way split
│   ├── environment.py        # hardware/runtime introspection
│   ├── metrics.py            # classification, ranking, operating point, bootstrap
│   ├── calibration.py        # Brier, ECE, reliability bins
│   ├── evaluation.py         # orchestration
│   ├── reporting.py          # plots + Markdown report
│   └── models/
│       ├── base.py           # interface: task, predict_one, predict_batch
│       ├── majority.py
│       ├── tfidf.py
│       ├── laya_model.py     # Laya zero-shot: choice/noul, N variants, batching
│       └── hatexplain.py
└── results/, report/         # generated, git-ignored
```

### Deviations from the PRD's proposed layout

- `src/models/laya.py` is named `laya_model.py`, so it can never shadow the installed
  `laya` package on import.
- The dataset is fetched as the authors' original CSV rather than via
  `huggingface/datasets`, so it can be hashed directly.
- The PRD's single `TF-IDF + LR` and single Laya prompt became **variants** (two weight
  settings; four Laya formulations), because comparing them separates a property of the
  model from a property of its training prior or its prompt.
- Fine-tuning and Detoxify are **not** implemented.

---

## Known gaps

| Area | Status |
| --- | --- |
| Dataset, grouped split, reproducibility | Done |
| Three-class metrics, confusion matrices, latency | Done |
| Hate-vs-rest with PR-AUC, bootstrap CIs | Done |
| Validation split; thresholds selected there | Done |
| Batched inference with separate throughput | Done |
| Binary formulations (choice, `noul`) | Done |
| Calibration measured (Brier, ECE, reliability) | Done |
| Temperature fitting / recalibration | Done (`results/calibration_<model>.json`) |
| Generalization to a second dataset | **Missing** |
| Qualitative error coding | Mechanical only |
| Fine-tuning Laya on the train split | **Missing (phase 2)** |

### Upstream issue found by this benchmark

`laya.calibrate.records_from_labeled` calls `agent._forward()` directly, but only
`predict` / `predict_batch` carry `@torch.no_grad()`. On a checkpoint whose parameters
require grad, building calibration records therefore fails with
`RuntimeError: Can't call numpy() on Tensor that requires grad` (laya 0.3.22).
`src/laya_calibration.py` wraps the call in `torch.no_grad()` as a behaviour-preserving
workaround; the fix belongs upstream.

### Recalibration

Laya's published checkpoints are over-confident, so the pipeline refits its
**temperature map** on the validation split, saves it as
`results/calibration_<model>.json`, and re-measures Brier and ECE on test. Temperature
scaling is monotone, so it cannot change a predicted label — the report asserts that
("labels changed" must be 0) instead of assuming it. Load the fitted map with:

```python
agent = laya.load("convaiinnovations/laya", calibration="results/calibration_laya.json")
```

---

## Repository contents, and what is deliberately left out

Tracked: the source, `config.yaml`, both READMEs, `docs/PRD.md`, the license, and a
small set of **aggregate** results (`results/metrics.json`, `model_comparison.csv`,
`operating_points.csv`, `coverage.csv`, `report/benchmark_report.md`).

Not tracked, and why:

| Path | Reason |
| --- | --- |
| `data/`, `results/predictions.csv`, `results/error_analysis.csv` | contain the **raw tweet text**, which is hate speech. They are regenerated by `python run_benchmark.py` and re-verified by SHA-256. |
| `.venv/`, `*.png` outputs beyond the report | regenerable |
| `results/summary.json` | regenerable; used only to re-render the report |

**Representative errors are redacted by default.** The report lists the error
categories and the example ids, but withholds the tweet text, because the report is
the artifact most likely to be published. Set
`output.error_examples_in_report: full` for a local report, and re-render without
re-running any model:

```bash
python scripts/render_report.py
```

---

## Publishing to GitHub

The virtual environment, the downloaded dataset and every generated artifact are
git-ignored, so a fresh clone stays small and rebuilds them with one command.

```bash
git init
git add .
git commit -m "feat: local hate speech benchmark comparing Laya formulations with baselines"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

To publish a specific generated result anyway:

```bash
git add -f report/benchmark_report.md results/model_comparison.csv
git commit -m "docs: add benchmark report for <date>"
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).
