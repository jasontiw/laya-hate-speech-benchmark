# Local Hate Speech Detection Benchmark — Technical Report

_Generated at 2026-09-29T21:25:50 by `run_benchmark.py`._

## 1. Executive Summary

This report benchmarks **8** classifier configurations on the Davidson hate-speech/offensive-language dataset: 6 three-class models and 2 binary (hate-vs-rest) variants. All were evaluated on the **same 4933 test tweets** from a single deterministic, duplicate-grouped split, with thresholds selected on a separate validation split.

Measured results (full tables in sections 7 and 8):

- **majority**: accuracy 77.54%, macro F1 0.2912, hate recall 0.0000, hate precision 0.0000, PR-AUC n/a.
- **tfidf**: accuracy 89.09%, macro F1 0.6579, hate recall 0.1373, hate precision 0.7222, PR-AUC 0.429.
- **tfidf_balanced**: accuracy 87.67%, macro F1 0.7449, hate recall 0.5810, hate precision 0.4015, PR-AUC 0.423.
- **laya**: accuracy 68.07%, macro F1 0.5653, hate recall 0.6725, hate precision 0.1880, PR-AUC 0.291.
- **laya_semantic**: accuracy 39.49%, macro F1 0.3651, hate recall 0.6549, hate precision 0.1616, PR-AUC 0.266.
- **laya_binary**: accuracy 76.75%, macro F1 n/a, hate recall 0.7183, hate precision 0.1605, PR-AUC 0.270.
- **laya_noul**: accuracy 53.46%, macro F1 n/a, hate recall 0.8697, hate precision 0.0986, PR-AUC 0.252.
- **hatexplain**: accuracy 61.16%, macro F1 0.4859, hate recall 0.2254, hate precision 0.3699, PR-AUC 0.260.

Highest macro F1 (three-class models): **tfidf_balanced** (0.7449).
Highest hate recall: **laya_noul** (0.8697).
Highest PR-AUC (hate-vs-rest): **tfidf** (0.429).

The models do not simply rank by quality: they **disagree about what counts as hate speech**. One is precise and rarely fires; another fires often and catches far more hate speech at a large precision cost. That behavioural difference, not a single F1, is the headline finding.

## 2. Research Question

How does Laya perform on a known hate-speech dataset compared with classical and specialised local classifiers, when all models are evaluated on the same examples with the same metrics? And, for Laya, which formulation (three-class choice, binary choice, noul, improved definitions) gives the best hate-recall / precision / cost trade-off?

## 3. Dataset

- **Name:** `tdavidson/hate_speech_offensive`
- **Source:** https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv
- **Description:** Davidson et al., "Automated Hate Speech Detection and the Problem of Offensive Language" (ICWSM 2017). 24,783 English tweets labeled into three classes.
- **Rows:** 24783
- **SHA-256:** `fcb8bc7c68120ae4af04a5b9acd58585513ede11e1548ebf36a5c2040b6f6281`
- **SHA-256 verified:** True

| Class | Rows |
| --- | ---: |
| hate speech | 1430 |
| offensive language | 19190 |
| neither | 4163 |

## 4. Experimental Setup

- **Split:** train / validation / test, stratified by class and grouped so duplicate tweets cannot cross a boundary. Seed `42`, test size `0.2`, validation size `0.15`.
- **Rows:** train 16130 | validation 3720 | test 4933
- **Test set signature (SHA-256 of test ids):** `be7f65da323d8f7f5db58d5597d6bf56fe107a817527c8343449674c65b17b12`
- **Groups (normalized tweets):** 24469
- **Hate speech in the test set:** 284 of 4933 (5.76%)

| Split | Hate speech | Offensive | Neither |
| --- | ---: | ---: | ---: |
| train | 933 | 12495 | 2702 |
| validation | 213 | 2870 | 637 |
| test | 284 | 3825 | 824 |

**Duplicate normalizer.** Duplicate detection normalizes the raw tweet in this order: lowercase; URLs removed; @mentions removed; a leading `rt` removed; punctuation optionally removed; whitespace collapsed and trimmed. Two tweets with the same normalized string share a group and are assigned to the same side of the split. The models always receive the raw, unmodified tweet.

**Threshold selection.** The threshold on P(hate speech) is chosen on the validation split and applied unchanged to the test set. The test-optimal threshold is recorded in `results/metrics.json` only as an upper bound, never as a headline number.

**Statistical robustness.** Every headline metric is reported with a 95% percentile bootstrap confidence interval over 1000 resamples of the test set (seed 42).

**Hardware / runtime** (PRD section 18):

- OS: Windows 11 (Windows-11-10.0.26200-SP0)
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD (24 cores)
- RAM: 34.28 GB
- Python: 3.12.10
- Packages: {"torch": "2.14.0+cu126", "transformers": "5.17.0", "laya": "0.3.22", "scikit_learn": "1.9.1", "pandas": "3.0.6", "numpy": "2.5.3"}
- CUDA available: True (torch CUDA 12.6)
- GPU: ['NVIDIA GeForce RTX 4060 Ti'] ([17.18] GB)

## 5. Models

| Model | Training regime | Trained on the Davidson train split? | What it is |
| --- | --- | --- | --- |
| majority | Baseline | No | Predicts the most frequent training label. |
| tfidf | Supervised (fit here) | Yes | TF-IDF (1-2 grams) + Logistic Regression, fit on the train split. |
| tfidf_balanced | Supervised (fit here) | Yes | Same pipeline with class_weight='balanced'. |
| laya | Zero-shot | No | Published checkpoint, hand-written question, no fine-tuning. |
| laya_semantic | Zero-shot | No | Published checkpoint, hand-written question, no fine-tuning. |
| laya_binary | Zero-shot | No | Published checkpoint, hand-written question, no fine-tuning. |
| laya_noul | Zero-shot | No | Published checkpoint, hand-written question, no fine-tuning. |
| hatexplain | Pretrained externally | No | HateXplain BERT, trained on the HateXplain dataset, labels remapped. |

> This benchmark compares end-to-end classification approaches under their natural training regimes; it is not a controlled architecture-vs-architecture comparison.

| Key | Model | Task | Device | Load (s) |
| --- | --- | --- | --- | ---: |
| majority | Majority baseline | three_class | n/a | 0.00 |
| tfidf | TF-IDF + Logistic Regression | three_class | n/a | 1.11 |
| tfidf_balanced | TF-IDF + Logistic Regression (class_weight=balanced) | three_class | n/a | 1.01 |
| laya | Laya (zero-shot, L1 3-class choice (PRD wording)) | three_class | cuda | 3.66 |
| laya_semantic | Laya (zero-shot, L4 3-class choice (explicit definitions)) | three_class | cuda | 1.25 |
| laya_binary | Laya (zero-shot, L2 2-class choice (hate / not hate)) | hate_binary | cuda | 1.06 |
| laya_noul | Laya (zero-shot, L3 2-class noul (is_hate)) | hate_binary | cuda | 0.92 |
| hatexplain | HateXplain BERT | three_class | cuda | 1.40 |

- Laya Hub commit: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
- HateXplain Hub commit: `e487c81b768c7532bf474bd5e486dedea4cf3848`

## 6. Evaluation Methodology

Each three-class model returns exactly one of hate speech / offensive language / neither. Each binary variant returns hate speech or not-hate-speech and is scored only on the binary problem. Every model sees exactly the same test rows, in the same order. Metrics use scikit-learn (`zero_division=0`) and hate-vs-rest PR-AUC (average precision), which is the fairer summary than ROC-AUC at ~5.8% positives.

**Inference.** Where the model has a batched path (Laya's `predict_batch`, HateXplain and TF-IDF), predictions come from one batched pass over the test set and single-item latency is measured separately on a sample, so throughput and latency are never conflated. The two paths are cross-checked: the report shows the label agreement between them.

## 7. Overall Results

### Three-class models

| Model | Accuracy | Macro F1 | Macro F1 95% CI | Hate Precision | Hate Recall | Hate F1 | PR-AUC | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 77.54% | 0.2912 | [0.2887, 0.2936] | 0.0000 | 0.0000 | 0.0000 | n/a | 0.0 ms |
| tfidf | 89.09% | 0.6579 | [0.6356, 0.6799] | 0.7222 | 0.1373 | 0.2308 | 0.429 | 0.4 ms |
| tfidf_balanced | 87.67% | 0.7449 | [0.7270, 0.7618] | 0.4015 | 0.5810 | 0.4748 | 0.423 | 0.4 ms |
| laya | 68.07% | 0.5653 | [0.5481, 0.5806] | 0.1880 | 0.6725 | 0.2938 | 0.291 | 37.7 ms |
| laya_semantic | 39.49% | 0.3651 | [0.3501, 0.3797] | 0.1616 | 0.6549 | 0.2592 | 0.266 | 37.2 ms |
| hatexplain | 61.16% | 0.4859 | [0.4646, 0.5056] | 0.3699 | 0.2254 | 0.2801 | 0.260 | 9.3 ms |

### Binary (hate-vs-rest) variants

| Model | Task | Accuracy (binary) | Hate Precision | Hate Recall | Hate F1 | PR-AUC | Hate FN | Hate FP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| laya_binary | hate_binary | 76.75% | 0.1605 | 0.7183 | 0.2624 | 0.270 | 80 | 1067 |
| laya_noul | hate_binary | 53.46% | 0.0986 | 0.8697 | 0.1771 | 0.252 | 37 | 2259 |

### Statistical robustness (95% bootstrap CI)

| Model | Accuracy | 95% CI | Macro F1 | 95% CI | Hate F1 | 95% CI | PR-AUC | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 0.7754 | [0.7636, 0.7868] | 0.2912 | [0.2887, 0.2936] | 0.0000 | [0.0000, 0.0000] | n/a | n/a |
| tfidf | 0.8909 | [0.8822, 0.8997] | 0.6579 | [0.6356, 0.6799] | 0.2308 | [0.1695, 0.2914] | 0.429 | [0.371, 0.494] |
| tfidf_balanced | 0.8767 | [0.8674, 0.8861] | 0.7449 | [0.7270, 0.7618] | 0.4748 | [0.4290, 0.5185] | 0.423 | [0.365, 0.487] |
| laya | 0.6807 | [0.6661, 0.6939] | 0.5653 | [0.5481, 0.5806] | 0.2938 | [0.2621, 0.3248] | 0.291 | [0.246, 0.346] |
| laya_semantic | 0.3949 | [0.3811, 0.4089] | 0.3651 | [0.3501, 0.3797] | 0.2592 | [0.2292, 0.2875] | 0.266 | [0.220, 0.325] |
| laya_binary | 0.7675 | [0.7559, 0.7788] | n/a | n/a | 0.2624 | [0.2368, 0.2914] | 0.270 | [0.227, 0.324] |
| laya_noul | 0.5346 | [0.5208, 0.5486] | n/a | n/a | 0.1771 | [0.1574, 0.1954] | 0.252 | [0.208, 0.304] |
| hatexplain | 0.6116 | [0.5982, 0.6254] | 0.4859 | [0.4646, 0.5056] | 0.2801 | [0.2269, 0.3313] | 0.260 | [0.215, 0.314] |

### Per-class metrics (three-class models)

**Majority baseline**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.0000 | 0.0000 | 0.0000 | 284 |
| offensive language | 0.7754 | 1.0000 | 0.8735 | 3825 |
| neither | 0.0000 | 0.0000 | 0.0000 | 824 |

**TF-IDF + Logistic Regression**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.7222 | 0.1373 | 0.2308 | 284 |
| offensive language | 0.8981 | 0.9752 | 0.9351 | 3825 |
| neither | 0.8623 | 0.7597 | 0.8077 | 824 |

**TF-IDF + Logistic Regression (class_weight=balanced)**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.4015 | 0.5810 | 0.4748 | 284 |
| offensive language | 0.9606 | 0.8915 | 0.9247 | 3825 |
| neither | 0.7716 | 0.9102 | 0.8352 | 824 |

**Laya (zero-shot, L1 3-class choice (PRD wording))**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.1880 | 0.6725 | 0.2938 | 284 |
| offensive language | 0.9329 | 0.6614 | 0.7741 | 3825 |
| neither | 0.5286 | 0.7731 | 0.6279 | 824 |

**Laya (zero-shot, L4 3-class choice (explicit definitions))**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.1616 | 0.6549 | 0.2592 | 284 |
| offensive language | 0.9338 | 0.2690 | 0.4177 | 3825 |
| neither | 0.2735 | 0.8896 | 0.4184 | 824 |

**HateXplain BERT**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| hate speech | 0.3699 | 0.2254 | 0.2801 | 284 |
| offensive language | 0.9102 | 0.5778 | 0.7069 | 3825 |
| neither | 0.3186 | 0.9017 | 0.4708 | 824 |

_Measured result. Interpreting which model is 'better' depends on whether false negatives or false positives on hate speech matter more for the intended use, which this benchmark does not decide._

## 8. Hate-Speech Results

| Model | Task | Hate precision | Hate recall | Hate F1 | PR-AUC | Hate FN | Hate FP | Hate support |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | three_class | 0.0000 | 0.0000 | 0.0000 | n/a | 284 | 0 | 284 |
| tfidf | three_class | 0.7222 | 0.1373 | 0.2308 | 0.429 | 245 | 15 | 284 |
| tfidf_balanced | three_class | 0.4015 | 0.5810 | 0.4748 | 0.423 | 119 | 246 | 284 |
| laya | three_class | 0.1880 | 0.6725 | 0.2938 | 0.291 | 93 | 825 | 284 |
| laya_semantic | three_class | 0.1616 | 0.6549 | 0.2592 | 0.266 | 98 | 965 | 284 |
| laya_binary | hate_binary | 0.1605 | 0.7183 | 0.2624 | 0.270 | 80 | 1067 | 284 |
| laya_noul | hate_binary | 0.0986 | 0.8697 | 0.1771 | 0.252 | 37 | 2259 | 284 |
| hatexplain | three_class | 0.3699 | 0.2254 | 0.2801 | 0.260 | 220 | 109 | 284 |

_Measured result. Hate recall = true hate tweets found / all true hate tweets. Hate FN are hate tweets labelled something else; hate FP are non-hate tweets labelled hate speech. Binary variants have no macro F1 because they never emit the other two labels._

### Operating point: natural decision vs a threshold on P(hate speech)

| Model | Decision rule | Accuracy | Macro F1 | Hate Precision | Hate Recall | Hate F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| majority | argmax (natural) | 77.54% | 0.2912 | 0.0000 | 0.0000 | 0.0000 |
| tfidf | argmax (natural) | 89.09% | 0.6579 | 0.7222 | 0.1373 | 0.2308 |
| tfidf | threshold P(hate) ≥ 0.13 (from validation) | 87.80% | 0.7358 | 0.4081 | 0.5704 | 0.4758 |
| tfidf_balanced | argmax (natural) | 87.67% | 0.7449 | 0.4015 | 0.5810 | 0.4748 |
| tfidf_balanced | threshold P(hate) ≥ 0.49 (from validation) | 88.06% | 0.7414 | 0.4360 | 0.5035 | 0.4673 |
| laya | argmax (natural) | 68.07% | 0.5653 | 0.1880 | 0.6725 | 0.2938 |
| laya | threshold P(hate) ≥ 0.61 (from validation) | 76.00% | 0.6009 | 0.2691 | 0.4225 | 0.3288 |
| laya_semantic | argmax (natural) | 39.49% | 0.3651 | 0.1616 | 0.6549 | 0.2592 |
| laya_semantic | threshold P(hate) ≥ 0.58 (from validation) | 49.32% | 0.4388 | 0.2911 | 0.4049 | 0.3387 |
| laya_binary | argmax (natural) | 76.75% | n/a | 0.1605 | 0.7183 | 0.2624 |
| laya_binary | threshold P(hate) ≥ 0.75 (from validation) | 90.29% | n/a | 0.2779 | 0.4296 | 0.3375 |
| laya_noul | argmax (natural) | 53.46% | n/a | 0.0986 | 0.8697 | 0.1771 |
| laya_noul | threshold P(hate) ≥ 0.75 (from validation) | 90.37% | n/a | 0.2742 | 0.4085 | 0.3281 |
| hatexplain | argmax (natural) | 61.16% | 0.4859 | 0.3699 | 0.2254 | 0.2801 |
| hatexplain | threshold P(hate) ≥ 0.12 (from validation) | 58.95% | 0.5034 | 0.2846 | 0.5070 | 0.3646 |

_Measured result. The threshold row uses the same scores as the row above it, so any difference is the decision rule, not the model. The threshold comes from the validation split; it was not tuned on the test set._

### Confidence, coverage and calibration (secondary)

| Model | Correct & high | Correct & low | Incorrect & high | Incorrect & low | Mean conf. correct | Mean conf. incorrect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 4299 | 96 | 443 | 95 | 0.8767 | 0.6648 |
| tfidf_balanced | 4159 | 166 | 445 | 163 | 0.7924 | 0.6251 |
| laya | 2303 | 1055 | 987 | 588 | 0.5851 | 0.5650 |
| laya_semantic | 1088 | 860 | 1786 | 1199 | 0.6038 | 0.5537 |
| laya_binary | 204 | 0 | 4729 | 0 | 0.7658 | 0.7600 |
| laya_noul | 247 | 0 | 4686 | 0 | 0.7369 | 0.6822 |
| hatexplain | 2751 | 266 | 1628 | 288 | 0.6253 | 0.5986 |

| Model | Base rate (val) | Brier | ECE | Bins |
| --- | ---: | ---: | ---: | ---: |
| tfidf | 0.0573 | 0.0443 | 0.0156 | 10 |
| tfidf_balanced | 0.0573 | 0.0595 | 0.1018 | 10 |
| laya | 0.0573 | 0.1180 | 0.2317 | 10 |
| laya_semantic | 0.0573 | 0.1090 | 0.2106 | 10 |
| laya_binary | 0.0573 | 0.1609 | 0.2810 | 10 |
| laya_noul | 0.0573 | 0.2565 | 0.4163 | 10 |
| hatexplain | 0.0573 | 0.0532 | 0.0327 | 10 |

_Measured on the **validation** split, not on test. Lower Brier and ECE are better. This measures calibration only; no temperature was fitted, so these are descriptive numbers for the shipped checkpoints._

**Abstention / coverage.** Dropping the least confident answers:

| Model | Confidence ≥ | Coverage | Accuracy on kept | Hate recall on kept |
| --- | ---: | ---: | ---: | ---: |
| tfidf | 0.00 | 100.00% | 89.09% | 13.73% |
| tfidf | 0.40 | 99.88% | 89.18% | 13.83% |
| tfidf | 0.50 | 96.13% | 90.66% | 11.38% |
| tfidf | 0.60 | 88.93% | 92.80% | 9.09% |
| tfidf | 0.70 | 81.47% | 94.97% | 5.52% |
| tfidf | 0.80 | 73.89% | 96.54% | 2.11% |
| tfidf | 0.90 | 58.06% | 97.66% | 0.00% |
| tfidf_balanced | 0.00 | 100.00% | 87.67% | 58.10% |
| tfidf_balanced | 0.40 | 99.09% | 88.18% | 58.70% |
| tfidf_balanced | 0.50 | 93.33% | 90.33% | 59.40% |
| tfidf_balanced | 0.60 | 84.23% | 92.49% | 58.33% |
| tfidf_balanced | 0.70 | 71.72% | 94.54% | 58.45% |
| tfidf_balanced | 0.80 | 50.68% | 95.68% | 65.96% |
| tfidf_balanced | 0.90 | 21.41% | 97.25% | 82.05% |
| laya | 0.00 | 100.00% | 68.07% | 67.25% |
| laya | 0.40 | 96.92% | 68.19% | 67.26% |
| laya | 0.50 | 66.69% | 70.00% | 76.02% |
| laya | 0.60 | 34.54% | 67.96% | 86.30% |
| laya | 0.70 | 17.31% | 70.26% | 92.71% |
| laya | 0.80 | 10.56% | 81.19% | 96.61% |
| laya | 0.90 | 3.32% | 95.12% | 100.00% |
| laya_semantic | 0.00 | 100.00% | 39.49% | 65.49% |
| laya_semantic | 0.40 | 90.15% | 38.93% | 68.05% |
| laya_semantic | 0.50 | 58.26% | 37.86% | 74.04% |
| laya_semantic | 0.60 | 35.88% | 43.73% | 77.54% |
| laya_semantic | 0.70 | 21.91% | 58.93% | 83.13% |
| laya_semantic | 0.80 | 12.85% | 78.71% | 91.67% |
| laya_semantic | 0.90 | 5.19% | 94.92% | 100.00% |
| laya_binary | 0.00 | 100.00% | 4.14% | 71.83% |
| laya_binary | 0.40 | 100.00% | 4.14% | 71.83% |
| laya_binary | 0.50 | 100.00% | 4.14% | 71.83% |
| laya_binary | 0.60 | 83.52% | 4.22% | 75.65% |
| laya_binary | 0.70 | 66.07% | 4.27% | 76.37% |
| laya_binary | 0.80 | 43.89% | 4.34% | 80.34% |
| laya_binary | 0.90 | 17.49% | 3.59% | 73.81% |
| laya_noul | 0.00 | 100.00% | 5.01% | 86.97% |
| laya_noul | 0.40 | 100.00% | 5.01% | 86.97% |
| laya_noul | 0.50 | 100.00% | 5.01% | 86.97% |
| laya_noul | 0.60 | 68.21% | 6.69% | 91.09% |
| laya_noul | 0.70 | 37.56% | 9.07% | 91.30% |
| laya_noul | 0.80 | 17.58% | 6.23% | 81.82% |
| laya_noul | 0.90 | 11.21% | 2.35% | 76.47% |
| hatexplain | 0.00 | 100.00% | 61.16% | 22.54% |
| hatexplain | 0.40 | 99.51% | 61.30% | 22.86% |
| hatexplain | 0.50 | 88.77% | 62.82% | 23.89% |
| hatexplain | 0.60 | 51.65% | 67.03% | 32.03% |
| hatexplain | 0.70 | 21.10% | 68.30% | 51.39% |
| hatexplain | 0.80 | 4.07% | 71.14% | 62.96% |
| hatexplain | 0.90 | 0.02% | 0.00% | n/a |

_Measured result. 'Correct & high/low' splits answers by whether the reported probability reached 0.50. Coverage is the fraction of answers kept above a confidence threshold; accuracy and hate recall are measured on the kept subset, so they describe a different population at each row and are not directly comparable across rows._

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
| hate speech | 39 | 227 | 18 |
| offensive language | 13 | 3730 | 82 |
| neither | 2 | 196 | 626 |

**TF-IDF + Logistic Regression (class_weight=balanced)**

| Actual \ Predicted | hate speech | offensive language | neither |
| --- | ---: | ---: | ---: |
| hate speech | 165 | 90 | 29 |
| offensive language | 222 | 3410 | 193 |
| neither | 24 | 50 | 750 |

**Laya (zero-shot, L1 3-class choice (PRD wording))**

| Actual \ Predicted | hate speech | offensive language | neither |
| --- | ---: | ---: | ---: |
| hate speech | 191 | 72 | 21 |
| offensive language | 748 | 2530 | 547 |
| neither | 77 | 110 | 637 |

**Laya (zero-shot, L4 3-class choice (explicit definitions))**

| Actual \ Predicted | hate speech | offensive language | neither |
| --- | ---: | ---: | ---: |
| hate speech | 186 | 49 | 49 |
| offensive language | 898 | 1029 | 1898 |
| neither | 67 | 24 | 733 |

**Laya (zero-shot, L2 2-class choice (hate / not hate))**

| Actual \ Predicted | hate speech | not hate speech |
| --- | ---: | ---: |
| hate speech | 3582 | 1067 |
| not hate speech | 80 | 204 |

**Laya (zero-shot, L3 2-class noul (is_hate))**

| Actual \ Predicted | hate speech | not hate speech |
| --- | ---: | ---: |
| hate speech | 2390 | 2259 |
| not hate speech | 37 | 247 |

**HateXplain BERT**

| Actual \ Predicted | hate speech | offensive language | neither |
| --- | ---: | ---: | ---: |
| hate speech | 64 | 154 | 66 |
| offensive language | 92 | 2210 | 1523 |
| neither | 17 | 64 | 743 |

Plots: `results/confusion_matrix_<model>.png` and `results/pr_curve_hate_vs_rest.png`.

## 10. Latency Results

| Model | Load (s) | Single-item p50 (ms) | Single-item p95 (ms) | Single-item (items/s) | Batch size | Batched total (s) | Batched (items/s) | Batch/single label agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| tfidf | 1.11 | 0.44 | 0.47 | 2231.50 | None | 0.07 | 71290.14 | 100.00% |
| tfidf_balanced | 1.01 | 0.44 | 0.49 | 2224.90 | None | 0.07 | 66796.03 | 100.00% |
| laya | 3.66 | 37.74 | 48.51 | 25.83 | 64 | 19.32 | 255.33 | 99.40% |
| laya_semantic | 1.25 | 37.25 | 44.37 | 26.18 | 64 | 23.51 | 209.84 | 99.00% |
| laya_binary | 1.06 | 37.91 | 48.02 | 25.62 | 64 | 17.85 | 276.30 | 99.80% |
| laya_noul | 0.92 | 36.94 | 47.57 | 26.18 | 64 | 12.95 | 380.92 | 99.80% |
| hatexplain | 1.40 | 9.33 | 13.10 | 100.82 | 64 | 7.12 | 693.03 | 100.00% |

_Load time is model loading/fitting. Single-item latency and throughput are measured on a latency-sample-row sample with a warm-up; the batched column is the full-test-set pass. They are reported separately because they measure different things._

### Token budget and truncation

| Model | Rows | Truncated rows | Mean state tokens | Max state tokens | Limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| laya | 4933 | 0 | 26.9 | 131 | 512 |
| laya_semantic | 4933 | 0 | 26.9 | 131 | 512 |
| laya_binary | 4933 | 0 | 26.9 | 131 | 512 |
| laya_noul | 4933 | 0 | 26.9 | 131 | 512 |
| hatexplain | 4933 | 6 | n/a | 128 | 128 |

## 11. Error Analysis

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
| laya_binary | hate speech -> not hate speech | 50 |
| laya_binary | neither -> hate speech | 50 |
| laya_binary | neither -> not hate speech | 50 |
| laya_binary | offensive language -> hate speech | 50 |
| laya_binary | offensive language -> not hate speech | 50 |
| laya_noul | neither -> hate speech | 50 |
| laya_noul | neither -> not hate speech | 50 |
| laya_noul | offensive language -> hate speech | 50 |
| laya_noul | offensive language -> not hate speech | 50 |
| laya_noul | hate speech -> not hate speech | 37 |
| laya_semantic | neither -> hate speech | 50 |
| laya_semantic | offensive language -> hate speech | 50 |
| laya_semantic | offensive language -> neither | 50 |
| laya_semantic | hate speech -> neither | 49 |
| laya_semantic | hate speech -> offensive language | 49 |
| laya_semantic | neither -> offensive language | 24 |
| majority | hate speech -> offensive language | 50 |
| majority | neither -> offensive language | 50 |
| tfidf | hate speech -> offensive language | 50 |
| tfidf | neither -> offensive language | 50 |
| tfidf | offensive language -> neither | 50 |
| tfidf | hate speech -> neither | 18 |
| tfidf | offensive language -> hate speech | 13 |
| tfidf | neither -> hate speech | 2 |
| tfidf_balanced | hate speech -> offensive language | 50 |
| tfidf_balanced | neither -> offensive language | 50 |
| tfidf_balanced | offensive language -> hate speech | 50 |
| tfidf_balanced | offensive language -> neither | 50 |
| tfidf_balanced | hate speech -> neither | 29 |
| tfidf_balanced | neither -> hate speech | 24 |

The full examples are in `results/error_analysis.csv`.

### Representative errors

The rows below are **selected automatically** — the most confident mistakes per
category, which are the most informative ones to read. They are candidates for
manual coding, not a qualitative conclusion.

> **Tweet text is withheld from this report.** The examples are hate speech; the verbatim rows are in `results/error_analysis.csv`, which is not version-controlled. Set `output.error_examples_in_report: full` to print them in a local report.

**hatexplain — hate speech -> neither (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | neither | 0.847 |
| n/a | hate speech | neither | 0.826 |
| n/a | hate speech | neither | 0.809 |

**hatexplain — hate speech -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | offensive language | 0.873 |
| n/a | hate speech | offensive language | 0.807 |
| n/a | hate speech | offensive language | 0.806 |

**hatexplain — neither -> hate speech (17 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | hate speech | 0.927 |
| n/a | neither | hate speech | 0.874 |
| n/a | neither | hate speech | 0.855 |

**hatexplain — neither -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | offensive language | 0.741 |
| n/a | neither | offensive language | 0.726 |
| n/a | neither | offensive language | 0.722 |

**hatexplain — offensive language -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | hate speech | 0.897 |
| n/a | offensive language | hate speech | 0.866 |
| n/a | offensive language | hate speech | 0.863 |

**hatexplain — offensive language -> neither (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | neither | 0.830 |
| n/a | offensive language | neither | 0.819 |
| n/a | offensive language | neither | 0.801 |

**laya — hate speech -> neither (21 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | neither | 0.865 |
| n/a | hate speech | neither | 0.847 |
| n/a | hate speech | neither | 0.750 |

**laya — hate speech -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | offensive language | 0.704 |
| n/a | hate speech | offensive language | 0.640 |
| n/a | hate speech | offensive language | 0.630 |

**laya — neither -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | hate speech | 0.891 |
| n/a | neither | hate speech | 0.871 |
| n/a | neither | hate speech | 0.854 |

**laya — neither -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | offensive language | 0.719 |
| n/a | neither | offensive language | 0.719 |
| n/a | neither | offensive language | 0.700 |

**laya — offensive language -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | hate speech | 0.911 |
| n/a | offensive language | hate speech | 0.874 |
| n/a | offensive language | hate speech | 0.857 |

**laya — offensive language -> neither (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | neither | 0.866 |
| n/a | offensive language | neither | 0.846 |
| n/a | offensive language | neither | 0.794 |

**laya_binary — hate speech -> not hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | not hate speech | 0.940 |
| n/a | hate speech | not hate speech | 0.933 |
| n/a | hate speech | not hate speech | 0.930 |

**laya_binary — neither -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | hate speech | 0.960 |
| n/a | neither | hate speech | 0.896 |
| n/a | neither | hate speech | 0.872 |

**laya_binary — neither -> not hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | not hate speech | 0.988 |
| n/a | neither | not hate speech | 0.984 |
| n/a | neither | not hate speech | 0.983 |

**laya_binary — offensive language -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | hate speech | 0.914 |
| n/a | offensive language | hate speech | 0.881 |
| n/a | offensive language | hate speech | 0.869 |

**laya_binary — offensive language -> not hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | not hate speech | 0.984 |
| n/a | offensive language | not hate speech | 0.955 |
| n/a | offensive language | not hate speech | 0.942 |

**laya_noul — hate speech -> not hate speech (37 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | not hate speech | 1.000 |
| n/a | hate speech | not hate speech | 0.998 |
| n/a | hate speech | not hate speech | 0.952 |

**laya_noul — neither -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | hate speech | 0.895 |
| n/a | neither | hate speech | 0.833 |
| n/a | neither | hate speech | 0.829 |

**laya_noul — neither -> not hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | not hate speech | 1.000 |
| n/a | neither | not hate speech | 1.000 |
| n/a | neither | not hate speech | 1.000 |

**laya_noul — offensive language -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | hate speech | 0.880 |
| n/a | offensive language | hate speech | 0.844 |
| n/a | offensive language | hate speech | 0.797 |

**laya_noul — offensive language -> not hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | not hate speech | 1.000 |
| n/a | offensive language | not hate speech | 1.000 |
| n/a | offensive language | not hate speech | 0.975 |

**laya_semantic — hate speech -> neither (49 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | neither | 0.851 |
| n/a | hate speech | neither | 0.848 |
| n/a | hate speech | neither | 0.836 |

**laya_semantic — hate speech -> offensive language (49 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | offensive language | 0.636 |
| n/a | hate speech | offensive language | 0.620 |
| n/a | hate speech | offensive language | 0.618 |

**laya_semantic — neither -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | hate speech | 0.905 |
| n/a | neither | hate speech | 0.853 |
| n/a | neither | hate speech | 0.850 |

**laya_semantic — neither -> offensive language (24 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | offensive language | 0.558 |
| n/a | neither | offensive language | 0.554 |
| n/a | neither | offensive language | 0.550 |

**laya_semantic — offensive language -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | hate speech | 0.790 |
| n/a | offensive language | hate speech | 0.782 |
| n/a | offensive language | hate speech | 0.730 |

**laya_semantic — offensive language -> neither (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | neither | 0.946 |
| n/a | offensive language | neither | 0.887 |
| n/a | offensive language | neither | 0.848 |

**majority — hate speech -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | offensive language | nan |
| n/a | hate speech | offensive language | nan |
| n/a | hate speech | offensive language | nan |

**majority — neither -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | offensive language | nan |
| n/a | neither | offensive language | nan |
| n/a | neither | offensive language | nan |

**tfidf — hate speech -> neither (18 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | neither | 0.899 |
| n/a | hate speech | neither | 0.775 |
| n/a | hate speech | neither | 0.724 |

**tfidf — hate speech -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | offensive language | 0.975 |
| n/a | hate speech | offensive language | 0.969 |
| n/a | hate speech | offensive language | 0.954 |

**tfidf — neither -> hate speech (2 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | hate speech | 0.396 |
| n/a | neither | hate speech | 0.363 |

**tfidf — neither -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | offensive language | 0.879 |
| n/a | neither | offensive language | 0.860 |
| n/a | neither | offensive language | 0.849 |

**tfidf — offensive language -> hate speech (13 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | hate speech | 0.813 |
| n/a | offensive language | hate speech | 0.743 |
| n/a | offensive language | hate speech | 0.645 |

**tfidf — offensive language -> neither (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | neither | 0.933 |
| n/a | offensive language | neither | 0.869 |
| n/a | offensive language | neither | 0.858 |

**tfidf_balanced — hate speech -> neither (29 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | neither | 0.945 |
| n/a | hate speech | neither | 0.851 |
| n/a | hate speech | neither | 0.825 |

**tfidf_balanced — hate speech -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | hate speech | offensive language | 0.960 |
| n/a | hate speech | offensive language | 0.924 |
| n/a | hate speech | offensive language | 0.912 |

**tfidf_balanced — neither -> hate speech (24 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | hate speech | 0.793 |
| n/a | neither | hate speech | 0.758 |
| n/a | neither | hate speech | 0.679 |

**tfidf_balanced — neither -> offensive language (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | neither | offensive language | 0.944 |
| n/a | neither | offensive language | 0.903 |
| n/a | neither | offensive language | 0.844 |

**tfidf_balanced — offensive language -> hate speech (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | hate speech | 0.963 |
| n/a | offensive language | hate speech | 0.939 |
| n/a | offensive language | hate speech | 0.936 |

**tfidf_balanced — offensive language -> neither (50 exported)**

| Example id | Gold | Predicted | Confidence |
| ---: | --- | --- | ---: |
| n/a | offensive language | neither | 0.974 |
| n/a | offensive language | neither | 0.927 |
| n/a | offensive language | neither | 0.916 |

([withheld — see results/error_analysis.csv])


## 12. Laya Analysis

Laya was run **zero-shot**; no fine-tuning was performed (PRD section 24 defers it to phase 2). Because a zero-shot model's answer depends on the prompt *and* on the question type, several formulations are compared — three-class choice, binary choice, noul, and an explicit-definition prompt:

| Variant | Task | Question | Prompt (instructions) | Accuracy | Macro F1 | Hate P | Hate R | Hate F1 | PR-AUC |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| laya | three_class | choice | Classify the language of this message. | 68.07% | 0.5653 | 0.1880 | 0.6725 | 0.2938 | 0.291 |
| laya_semantic | three_class | choice | Classify the tweet into exactly one category. | 39.49% | 0.3651 | 0.1616 | 0.6549 | 0.2592 | 0.266 |
| laya_binary | hate_binary | choice | Is this text hate speech, or not hate speech? | 76.75% | n/a | 0.1605 | 0.7183 | 0.2624 | 0.270 |
| laya_noul | hate_binary | noul | Is this text hate speech? | 53.46% | n/a | 0.0986 | 0.8697 | 0.1771 | 0.252 |

_Prompt/formulation sensitivity (measured): across 4 Laya variants, hate recall spans 0.6549 to 0.8697 on the same test rows._

_No variant was selected using the test set: all formulations are reported, and thresholds come from validation. Choosing a single formulation for production is a separate decision that needs its own validation protocol._

## 13. Limitations

- Single dataset (Davidson), single language (English), single split seed.
- **No generalization test.** A strong result on one dataset does not establish generalization; a second hate-speech dataset is required.
- **Calibration is measured, not fitted.** Brier and ECE are reported for the shipped checkpoints; no temperature scaling was applied, so the confidence numbers are descriptive and should not be read as 'about c of answers at c are correct'.
- **No fine-tuning.** Laya's zero-shot numbers are a starting point, not its ceiling.
- Models have different label taxonomies: HateXplain's own three classes are mapped onto Davidson's, which introduces a mapping assumption the report cannot remove.
- The benchmark compares end-to-end approaches under their natural training regimes (see section 5), so it is not evidence that one architecture is better than another.
- Latency was measured per item on this machine and device only; it does not transfer to other hardware. Batched and single-item numbers are both reported, and batched shapes can change floating-point results, so their label agreement is reported too.
- The Davidson labels are themselves noisy and imbalanced; the majority class is 'offensive language', so accuracy is dominated by that class.

### Known gaps

| Area | Status |
| --- | --- |
| Dataset, grouped split, reproducibility | Done |
| Three-class metrics, confusion matrices, latency | Done |
| Hate-vs-rest with PR-AUC, bootstrap CIs | Done |
| Validation split; thresholds selected there | Done |
| Batched inference with separate throughput | Done |
| Binary formulations (choice, noul) | Done |
| Calibration measured (Brier, ECE, reliability) | Done |
| Temperature fitting / recalibration | **Missing** |
| Generalization to a second dataset | **Missing** |
| Qualitative error coding | Mechanical only (see section 11) |
| Fine-tuning Laya on the train split | **Missing (phase 2)** |

## 14. Conclusions

**Measured result.**

- **majority**: accuracy 77.54%, macro F1 0.2912, hate recall 0.0000, hate precision 0.0000, PR-AUC n/a.
- **tfidf**: accuracy 89.09%, macro F1 0.6579, hate recall 0.1373, hate precision 0.7222, PR-AUC 0.429.
- **tfidf_balanced**: accuracy 87.67%, macro F1 0.7449, hate recall 0.5810, hate precision 0.4015, PR-AUC 0.423.
- **laya**: accuracy 68.07%, macro F1 0.5653, hate recall 0.6725, hate precision 0.1880, PR-AUC 0.291.
- **laya_semantic**: accuracy 39.49%, macro F1 0.3651, hate recall 0.6549, hate precision 0.1616, PR-AUC 0.266.
- **laya_binary**: accuracy 76.75%, macro F1 n/a, hate recall 0.7183, hate precision 0.1605, PR-AUC 0.270.
- **laya_noul**: accuracy 53.46%, macro F1 n/a, hate recall 0.8697, hate precision 0.0986, PR-AUC 0.252.
- **hatexplain**: accuracy 61.16%, macro F1 0.4859, hate recall 0.2254, hate precision 0.3699, PR-AUC 0.260.

**Interpretation.**

Read the table as a whole rather than by a single metric. `tfidf_balanced` leads on macro F1 among the three-class models, but its hate recall is 0.5810: a precise, conservative detector. `laya_noul` has the highest hate recall (0.8697), i.e. a sensitive, over-triggering detector. `majority` is the most conservative of all. These are not interchangeable, and which is 'better' is a product decision about false negatives vs false positives, not a property of the models.

For Laya specifically, the interesting question is not whether it beats a supervised baseline, but **which formulation** gives the best recall/precision/cost trade-off — section 12 measures that, and section 8 shows how much the decision rule alone changes the answer.

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

