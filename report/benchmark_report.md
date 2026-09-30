# Local Hate Speech Detection Benchmark — Technical Report

_Generated at 2026-09-29T20:51:41 by `run_benchmark.py`._

## 1. Executive Summary

This report benchmarks **4** classifiers on the Davidson hate-speech/offensive-language dataset. All comparable models were evaluated on the **same 4933 test tweets** produced by a single deterministic, duplicate-grouped split.

Measured results (see section 7 for the full table):

- **majority**: accuracy 77.54%, macro F1 0.2912, hate recall 0.0000, hate precision 0.0000.
- **tfidf**: accuracy 89.24%, macro F1 0.6658, hate recall 0.1514, hate precision 0.7288.
- **laya**: accuracy 67.79%, macro F1 0.5628, hate recall 0.6725, hate precision 0.1869.
- **hatexplain**: accuracy 61.16%, macro F1 0.4859, hate recall 0.2254, hate precision 0.3699.

On this test set the highest macro F1 was **tfidf** (0.6658). The majority baseline is a reminder that accuracy alone is uninformative on this imbalanced dataset.

## 2. Research Question

How does Laya perform on a known hate-speech dataset compared with classical and specialised local classifiers, when all models are evaluated on the same examples with the same metrics?

## 3. Dataset

- **Name:** `tdavidson/hate_speech_offensive`
- **Source:** https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv
- **Description:** Davidson et al., "Automated Hate Speech Detection and the Problem of Offensive Language" (ICWSM 2017). 24,783 English tweets labeled into three classes.
- **Rows:** 24783
- **SHA-256:** `fcb8bc7c68120ae4af04a5b9acd58585513ede11e1548ebf36a5c2040b6f6281`
- **SHA-256 verified:** True
- **Loaded at:** 2026-09-29T20:51:46

| Class | Rows |
| --- | ---: |
| hate speech | 1430 |
| offensive language | 19190 |
| neither | 4163 |

## 4. Experimental Setup

- **Split:** stratified by class, grouped so that duplicate tweets cannot cross the boundary. Seed `42`, test size `0.2`.
- **Train rows:** 19850  |  **Test rows:** 4933
- **Test set signature (SHA-256 of test ids):** `be7f65da323d8f7f5db58d5597d6bf56fe107a817527c8343449674c65b17b12`
- **Groups (normalized tweets):** 24469

| Split | Hate speech | Offensive | Neither |
| --- | ---: | ---: | ---: |
| train | 1146 | 15365 | 3339 |
| test | 284 | 3825 | 824 |

**Hardware / runtime** (PRD section 18):

- OS: Windows 11 (Windows-11-10.0.26200-SP0)
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD (24 cores)
- RAM: 34.28 GB
- Python: 3.12.10
- Packages: {"torch": "2.14.0+cu126", "transformers": "5.17.0", "laya": "0.3.22", "scikit_learn": "1.9.1", "pandas": "3.0.6", "numpy": "2.5.3"}
- CUDA available: True (torch CUDA 12.6)
- GPU: ['NVIDIA GeForce RTX 4060 Ti'] ([17.18] GB)

## 5. Models

| Key | Model | Device | Load (s) |
| --- | --- | --- | ---: |
| majority | Majority baseline | n/a | 0.00 |
| tfidf | TF-IDF + Logistic Regression | n/a | 1.29 |
| laya | Laya (zero-shot) | cuda | 4.75 |
| hatexplain | HateXplain BERT | cuda | 1.43 |

- Laya Hub commit: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
- HateXplain Hub commit: `e487c81b768c7532bf474bd5e486dedea4cf3848`
- Laya checkpoint requested: `convaiinnovations/laya` (resolved: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`)

## 6. Evaluation Methodology

Each model receives the raw tweet text and returns exactly one of the three canonical labels. Every model sees exactly the same test rows, in the same order. Metrics are computed with scikit-learn (`zero_division=0`). Hate speech is additionally treated as a binary problem (hate vs rest) to expose its precision, recall and error counts, which aggregate accuracy hides.

## 7. Overall Results

| Model | Accuracy | Macro F1 | Hate Precision | Hate Recall | Hate F1 | Hate FN | Hate FP | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 77.54% | 0.2912 | 0.0000 | 0.0000 | 0.0000 | 284 | 0 | 0.0 ms | 0.0 ms |
| tfidf | 89.24% | 0.6658 | 0.7288 | 0.1514 | 0.2507 | 241 | 16 | 0.5 ms | 0.6 ms |
| laya | 67.79% | 0.5628 | 0.1869 | 0.6725 | 0.2925 | 93 | 831 | 36.6 ms | 43.9 ms |
| hatexplain | 61.16% | 0.4859 | 0.3699 | 0.2254 | 0.2801 | 220 | 109 | 8.6 ms | 10.3 ms |

### Per-class metrics

**Majority baseline**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.0000 | 0.0000 | 0.0000 | 284 |
| offensive language | 0.7754 | 1.0000 | 0.8735 | 3825 |
| neither | 0.0000 | 0.0000 | 0.0000 | 824 |

**TF-IDF + Logistic Regression**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.7288 | 0.1514 | 0.2507 | 284 |
| offensive language | 0.8999 | 0.9749 | 0.9359 | 3825 |
| neither | 0.8630 | 0.7646 | 0.8108 | 824 |

**Laya (zero-shot)**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.1869 | 0.6725 | 0.2925 | 284 |
| offensive language | 0.9316 | 0.6586 | 0.7716 | 3825 |
| neither | 0.5253 | 0.7694 | 0.6243 | 824 |

**HateXplain BERT**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.3699 | 0.2254 | 0.2801 | 284 |
| offensive language | 0.9102 | 0.5778 | 0.7069 | 3825 |
| neither | 0.3186 | 0.9017 | 0.4708 | 824 |

_Measured result. Interpreting which model is 'better' depends on whether false negatives or false positives on hate speech matter more for the intended use, which this benchmark does not decide._

## 8. Hate-Speech Results

| Model | Hate precision | Hate recall | Hate F1 | Hate FN | Hate FP | Hate support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 0.0000 | 0.0000 | 0.0000 | 284 | 0 | 284 |
| tfidf | 0.7288 | 0.1514 | 0.2507 | 241 | 16 | 284 |
| laya | 0.1869 | 0.6725 | 0.2925 | 93 | 831 | 284 |
| hatexplain | 0.3699 | 0.2254 | 0.2801 | 220 | 109 | 284 |

_Measured result. Hate recall = true hate tweets found / all true hate tweets. Hate FN are hate tweets labelled something else; hate FP are non-hate tweets labelled hate speech._

## 9. Confusion Matrices

**Majority baseline**

| Actual \ Predicted | hate speech | offensive language | neither |
| --- | ---: | ---: | ---: |
| hate speech | 0 | 284 | 0 |
| offensive language | 0 | 3825 | 0 |
| neither | 0 | 824 | 0 |

**TF-IDF + Logistic Regression**

| Actual \ Predicted | hate speech | offensive language | neither |
| --- | ---: | ---: | ---: |
| hate speech | 43 | 222 | 19 |
| offensive language | 15 | 3729 | 81 |
| neither | 1 | 193 | 630 |

**Laya (zero-shot)**

| Actual \ Predicted | hate speech | offensive language | neither |
| --- | ---: | ---: | ---: |
| hate speech | 191 | 72 | 21 |
| offensive language | 754 | 2519 | 552 |
| neither | 77 | 113 | 634 |

**HateXplain BERT**

| Actual \ Predicted | hate speech | offensive language | neither |
| --- | ---: | ---: | ---: |
| hate speech | 64 | 154 | 66 |
| offensive language | 92 | 2210 | 1523 |
| neither | 17 | 64 | 743 |

Plots are written to `results/confusion_matrix_<model>.png`.

## 10. Latency Results

| Model | Load (s) | Total (s) | Mean (ms) | p50 (ms) | p95 (ms) | Throughput (items/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a |
| tfidf | 1.29 | 2.29 | 0.47 | 0.45 | 0.57 | 2148.99 |
| laya | 4.75 | 183.47 | 37.19 | 36.56 | 43.86 | 26.89 |
| hatexplain | 1.43 | 43.12 | 8.74 | 8.60 | 10.29 | 114.40 |

_Load time is model loading/fitting and is reported separately from inference (PRD section 17). Latency measures warm per-item inference only._

## 11. Error Analysis

Counts of exported misclassification categories (capped per category):

| Model | Category | Exported |
| --- | --- | ---: |
| hatexplain | hate speech -> neither | 50 |
| hatexplain | hate speech -> offensive language | 50 |
| hatexplain | neither -> offensive language | 50 |
| hatexplain | offensive language -> hate speech | 50 |
| hatexplain | offensive language -> neither | 50 |
| hatexplain | neither -> hate speech | 17 |
| laya | hate speech -> offensive language | 50 |
| laya | neither -> hate speech | 50 |
| laya | neither -> offensive language | 50 |
| laya | offensive language -> hate speech | 50 |
| laya | offensive language -> neither | 50 |
| laya | hate speech -> neither | 21 |
| majority | hate speech -> offensive language | 50 |
| majority | neither -> offensive language | 50 |
| tfidf | hate speech -> offensive language | 50 |
| tfidf | neither -> offensive language | 50 |
| tfidf | offensive language -> neither | 50 |
| tfidf | hate speech -> neither | 19 |
| tfidf | offensive language -> hate speech | 15 |
| tfidf | neither -> hate speech | 1 |

The full examples are in `results/error_analysis.csv`.

## 12. Laya Analysis

Laya was run **zero-shot** as a `choice` question; no fine-tuning was performed in this MVP (PRD section 24 defers it to phase 2). Measured on the shared test set:

- Accuracy 67.79%, macro F1 0.5628.
- Hate speech: precision 0.1869, recall 0.6725, F1 0.2925 (93 false negatives, 831 false positives).
- Per class recall: hate 0.6725, offensive 0.6586, neither 0.7694.
- Inference: p50 36.56 ms, p95 43.86 ms, throughput 26.89 items/s.

### Confidence (secondary)

| Model | Mean confidence (correct) | Mean confidence (incorrect) | Mean confidence (all) |
| --- | ---: | ---: | ---: |
| laya | 0.5859 | 0.5636 | 0.5787 |

_Secondary analysis (PRD section 23). 'Confidence' here is the probability of the reported answer; it is not a calibration curve, and it is not comparable across models with different label sets or temperatures._

## 13. Limitations

- Single dataset (Davidson), single language (English), single split seed.
- A strong result on one dataset does not establish generalisation; cross-dataset testing is a phase-2 item (PRD section 30).
- Models have different label taxonomies: HateXplain's own three classes are mapped onto Davidson's, which introduces a mapping assumption the report cannot remove.
- Latency was measured per item on this machine and device only; it does not transfer to other hardware.
- The Davidson labels are themselves noisy and imbalanced; the majority class is 'offensive language', so accuracy is dominated by that class.
- This is a research benchmark, not a moderation system; no threshold tuning, calibration fitting or deployment policy was applied.

## 14. Conclusions

**Measured result.**

- **majority**: accuracy 77.54%, macro F1 0.2912, hate recall 0.0000, hate precision 0.0000.
- **tfidf**: accuracy 89.24%, macro F1 0.6658, hate recall 0.1514, hate precision 0.7288.
- **laya**: accuracy 67.79%, macro F1 0.5628, hate recall 0.6725, hate precision 0.1869.
- **hatexplain**: accuracy 61.16%, macro F1 0.4859, hate recall 0.2254, hate precision 0.3699.

**Interpretation.**

Reading the table as a whole rather than by any single metric: `tfidf` has the highest macro F1 here, while hate-speech recall separates the models far more than accuracy does. The majority baseline shows why accuracy alone would be misleading, and Laya's zero-shot behaviour should be read as a starting point that fine-tuning (phase 2) is meant to change, not as its ceiling.

## 15. Reproduction Instructions

```bash
# 1. Environment (PyTorch build first, matching your hardware)
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt

# 2. Run everything
python run_benchmark.py
```

The split is reproducible: the same seed and normalization reproduce the same test ids, and section 4 records the SHA-256 of the test id list. Pinned model revisions are in `results/experiment_config.json`.

## 16. References

1. Davidson, T., Warmsley, D., Macy, M., Weber, I. *Automated Hate Speech Detection and the Problem of Offensive Language.* ICWSM 2017.
2. Davidson hate-speech/offensive-language dataset, 24,783 English tweets, three classes.
3. Laya decision-engine repository and documentation (Convai Innovations).
4. Mathew, B. et al. *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*; model `Hate-speech-CNERG/bert-base-uncased-hatexplain`.

