# Local Hate Speech Detection Benchmark

**English** | [Español](README.es.md)

A small, local, reproducible experiment that measures how **Laya** performs on
hate-speech detection and compares it against classical and specialised classifiers
on the **same public dataset** and the **same test rows**.

Research question:

> How does Laya perform on a known hate-speech dataset compared with other local
> classification approaches when all models are evaluated on the same examples and
> with the same metrics?

This is a research benchmark, **not** a production moderation system. It reduces to
one command and a set of generated artifacts.

---

## Models compared

| Key | Model | Trained here? | Probabilities |
| --- | --- | --- | --- |
| `majority` | Majority-class baseline | fit on train split | no |
| `tfidf` | TF-IDF + Logistic Regression | fit on train split | yes |
| `laya` | Laya (zero-shot `choice` question) | no | yes |
| `hatexplain` | `Hate-speech-CNERG/bert-base-uncased-hatexplain` | no | yes |

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

On macOS/Linux use `.venv/bin/python run_benchmark.py`, or simply `python` inside an
activated virtual environment.

The first run downloads the dataset (~2.5 MB) and the model checkpoints
(Laya English, HateXplain BERT); those are cached by Hugging Face afterwards.

### Useful flags

```bash
python run_benchmark.py --limit 200                  # fast dev smoke run (clearly marked "not a benchmark result")
python run_benchmark.py --models tfidf,laya          # only some models
python run_benchmark.py --config my_experiment.yaml  # alternative config
python run_benchmark.py --list-models                # print model keys and exit
```

Every run is driven by [`config.yaml`](config.yaml); that file is copied verbatim
into `results/experiment_config.json`.

---

## Output artifacts

```
results/
├── dataset_metadata.json     # source, SHA-256, rows, class distribution
├── experiment_config.json    # config + hardware/runtime + resolved model revisions
├── split.json                # seed, counts, and the exact test id list
├── predictions.csv           # one row per test tweet, all model predictions + probabilities
├── metrics.json              # full metrics per model
├── model_comparison.csv      # the PRD section 21 comparison table
├── error_analysis.csv        # misclassified examples by (gold -> predicted) category
├── confusion_matrix_<model>.png
└── report.md

report/
└── benchmark_report.md       # the final deliverable
```

`benchmark_report.md` has the 16 sections required by the PRD (executive summary,
dataset, setup, models, methodology, overall and hate-specific results, confusion
matrices, latency, error analysis, Laya analysis, limitations, conclusions,
reproduction instructions, references).

---

## Methodology

- **Dataset handling.** Downloaded programmatically, SHA-256 verified, never edited,
  original labels preserved.
- **Split.** Fixed stratified 80/20 split with a deterministic seed. Tweets that
  normalize to the same string are kept in the same split, so a duplicated tweet
  cannot leak from train to test. Normalization is used *only* to detect duplicates —
  models always see the raw tweet text.
- **Laya question.** Hate-speech detection is posed as a Laya `choice` question with
  the three labels and their descriptions. No fine-tuning.
- **HateXplain mapping.** Its `id2label` is read from the model config and mapped
  onto the canonical labels (`hate speech -> hate speech`, `offensive -> offensive
  language`, `normal -> neither`). The raw mapping is recorded in the report.
- **Metrics.** Accuracy, macro F1, macro precision/recall, per-class precision/recall/F1,
  confusion matrices, and hate-specific precision/recall/F1 with false-positive and
  false-negative counts.
- **Latency.** Per-item inference time with a warm-up, reported as total, mean, p50,
  p95 and throughput. Model load/`fit` time is reported separately.
- **Reproducibility.** Same config → same split (the SHA-256 of the test id list is
  recorded). Pinned model revisions can be set in `config.yaml`.

---

## Project layout

```
laya-hate-speech-benchmark/
├── config.yaml               # experiment settings (single source of truth)
├── run_benchmark.py          # entry point: python run_benchmark.py
├── requirements.txt
├── pyproject.toml
├── data/{raw,processed}/     # git-ignored
├── scripts/run_benchmark.py  # thin wrapper (same entry point)
├── src/
│   ├── config.py             # typed config + canonical labels
│   ├── dataset.py            # download, verify, normalize, grouped split
│   ├── environment.py        # hardware/runtime introspection
│   ├── metrics.py            # classification + latency metrics
│   ├── evaluation.py         # orchestration
│   ├── reporting.py          # confusion-matrix plots + Markdown report
│   └── models/
│       ├── base.py           # common interface
│       ├── majority.py
│       ├── tfidf.py
│       ├── laya_model.py     # Laya zero-shot
│       └── hatexplain.py
└── results/, report/         # generated, git-ignored
```

### Deviations from the PRD's proposed layout

- `src/models/laya.py` is named `laya_model.py`. A module literally named `laya.py`
  inside this package risks shadowing the installed `laya` package on import; the
  renamed file removes that ambiguity. Everything else follows the proposed layout.
- The dataset is fetched as the authors' original CSV rather than via
  `huggingface/datasets`, so the downloaded artifact can be hashed directly.
- Fine-tuning, Detoxify and the phase-2 extensions are **not** implemented: this is
  the MVP only.

---

## Acceptance criteria

| AC | Where it is satisfied |
| --- | --- |
| AC1 dataset downloaded + verified | `src/dataset.py`, `results/dataset_metadata.json` |
| AC2 deterministic split persisted | `src/dataset.py`, `results/split.json` |
| AC3 Laya runs the full test set | `src/models/laya_model.py` |
| AC4 TF-IDF + LR predictions | `src/models/tfidf.py` |
| AC5 HateXplain predictions | `src/models/hatexplain.py` |
| AC6 all four models, all metrics | `src/metrics.py`, `results/metrics.json` |
| AC7 hate precision/recall/F1 reported | report section 8 |
| AC8 confusion matrix per model | `results/confusion_matrix_*.png` |
| AC9 latency recorded | `results/metrics.json` (`latency`), report section 10 |
| AC10 all predictions in one file | `results/predictions.csv` |
| AC11 Markdown report generated | `report/benchmark_report.md` |
| AC12 reproducible run | fixed seed + recorded test-id SHA-256 |

---

## Publishing to GitHub

The project is self-contained. The virtual environment, the downloaded dataset and
every generated artifact are git-ignored, so a fresh clone stays small and rebuilds
them with one command.

```bash
git init
git add .
git commit -m "feat: local hate speech benchmark comparing Laya with baselines"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

To publish a specific generated result anyway (for example the report of a real run):

```bash
git add -f report/benchmark_report.md results/model_comparison.csv
git commit -m "docs: add benchmark report for <date>"
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).
