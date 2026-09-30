# PRD — Local Hate Speech Detection Benchmark with Laya

## 1. Overview

### Project name

**Local Hate Speech Detection Benchmark**

### Project type

Small-scale Machine Learning / NLP Proof of Concept.

### Objective

Build a local, reproducible experiment to evaluate how well **Laya** performs on hate-speech detection and compare its results against other existing approaches using the **same public labeled dataset**.

The project is not intended to build a production moderation system. Its purpose is to answer a much simpler research question:

> **How does Laya perform on a known hate-speech dataset compared with other local classification approaches when all models are evaluated on the same examples and with the same metrics?**

The final result should be a reproducible set of predictions, metrics, visualizations, and a technical report.

---

# 2. Background

Laya is a non-autoregressive decision/classification model that accepts a text and a predefined `choice` question, returning a decision over the available labels. This makes it possible to formulate hate-speech detection as a three-class classification problem.

A recent independent experiment evaluated Laya using the Davidson hate-speech dataset containing **24,783 English tweets** with three labels:

* `hate speech`
* `offensive language`
* `neither`

The experiment reported the following results for its specific protocol:

| Model             | Accuracy | Macro F1 | Hate-speech Recall |
| ----------------- | -------: | -------: | -----------------: |
| Majority baseline |   77.43% |   0.2909 |              0.00% |
| Original Laya     |   22.10% |   0.2228 |             77.97% |
| Fine-tuned Laya   |   82.75% |   0.6521 |             41.43% |

The experiment also showed that fine-tuning substantially changed the precision/recall tradeoff for hate speech. Importantly, the author explicitly noted that the work did **not** establish a comparison against Jev or a simpler conventional classification model.

This PoC will reproduce the spirit of that experiment while adding independent comparison models.

---

# 3. Problem Statement

A model can achieve high overall accuracy on this dataset simply by predicting the most common class.

Therefore, evaluating only `accuracy` is insufficient.

The experiment must determine:

1. How frequently each model correctly identifies hate speech.
2. How often offensive language is confused with hate speech.
3. How often legitimate/non-hateful text is incorrectly flagged.
4. Whether a model's overall performance hides poor performance on the minority hate-speech class.
5. How expensive each approach is to run locally in terms of inference time and resources.

---

# 4. Goals

## Primary goals

### G1 — Run Laya locally

Install and execute a pinned version/checkpoint of Laya locally.

### G2 — Evaluate Laya on a public dataset

Run Laya against a fixed test set containing labeled examples.

### G3 — Establish comparable baselines

Run at least two additional approaches:

1. **TF-IDF + Logistic Regression**
2. **HateXplain BERT classifier**

HateXplain is particularly useful because its model outputs the same conceptual three-way classification:

* Hatespeech
* Offensive
* Normal

and is available as a local Hugging Face model.

### G4 — Produce comparable metrics

All comparable models must be evaluated against exactly the same test examples.

### G5 — Produce a technical report

Generate an automated report containing:

* dataset information
* experiment configuration
* model configuration
* overall metrics
* per-class metrics
* confusion matrices
* hate-speech precision/recall
* inference performance
* representative errors
* conclusions and limitations

---

# 5. Non-Goals

The MVP must deliberately avoid becoming a large ML platform.

The following are out of scope:

* Production moderation API
* Web application
* Model serving infrastructure
* Model training pipelines for multiple models
* Distributed inference
* Continuous monitoring
* Data annotation platform
* Automatic content blocking
* Real-time moderation
* Multilingual benchmarking
* Custom neural-network architecture
* Large-scale hyperparameter optimization

The project should remain a **research/benchmarking script with reproducible outputs**.

---

# 6. Dataset

## Primary dataset

**Davidson — Automated Hate Speech Detection and the Problem of Offensive Language**

Dataset:

`tdavidson/hate_speech_offensive`

It contains **24,783 English tweets** and three classes:

| ID | Label              |
| -: | ------------------ |
|  0 | hate speech        |
|  1 | offensive language |
|  2 | neither            |

The dataset is specifically designed to distinguish hate speech from offensive language, which makes it appropriate for this experiment.

The original repository is associated with the ICWSM 2017 paper and contains the labeled CSV dataset.

## Dataset handling requirements

The experiment must:

* Download the dataset programmatically.
* Record the dataset version/source.
* Record a SHA-256 hash of the downloaded file.
* Never manually modify the dataset.
* Preserve the original labels.
* Store the exact train/test split used.
* Use deterministic random seeds.

---

# 7. Dataset Split

For the MVP, use a fixed stratified split.

Recommended:

* **80% training**
* **20% test**

The split must be stratified by class.

Because duplicated or near-duplicated tweets can create leakage, normalized duplicate tweets should remain in the same split.

Normalization should be limited to identifying duplicates; the original tweet text should still be supplied to the models.

Example:

```text
raw tweet
    ↓
normalization
    ↓
duplicate/group detection
    ↓
stratified split by group
    ↓
train / test
```

The test set must never be used for training or model selection.

---

# 8. Models Under Test

## Model A — Majority baseline

Purpose:

Establish the minimum reference point.

The classifier always predicts the most frequent training class.

This is important because the Davidson dataset is imbalanced, meaning a high accuracy number can be misleading.

Expected behavior:

```text
prediction = most_common_training_label
```

---

# 9. Model B — TF-IDF + Logistic Regression

Purpose:

Provide a simple classical Machine Learning baseline.

Pipeline:

```text
Tweet
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
3-class prediction
```

Suggested implementation:

```python
TfidfVectorizer(...)
LogisticRegression(...)
```

This model will be trained exclusively using the training split.

This gives the experiment an important reference point:

> How much value does a modern decision model provide compared with a relatively simple text classifier?

---

# 10. Model C — Laya

## Base Laya

Run the original/published Laya checkpoint without fine-tuning.

The classification question should explicitly define the three labels:

```text
Classify the language of this message.

Hate speech:
language attacking or expressing hostility toward a protected group.

Offensive language:
rude, insulting, profane, or abusive language that does not constitute hate speech.

Neither:
language that does not belong to either category.
```

Options:

```text
hate speech
offensive language
neither
```

This is important because Laya evaluates the question and candidate labels supplied by the application.

## Laya result

The experiment must capture:

```text
predicted_label
probability_hate
probability_offensive
probability_neither
latency_ms
```

The exact model/checkpoint revision must be pinned and recorded.

---

# 11. Model D — HateXplain BERT

Use:

`Hate-speech-CNERG/bert-base-uncased-hatexplain`

The model is designed for three-way classification:

```text
Hatespeech
Offensive
Normal
```

and is available for local inference through Hugging Face Transformers.

Pipeline:

```text
Tweet
 ↓
BERT classifier
 ↓
3-class prediction
```

The output labels must be mapped into the project's canonical labels:

```text
Hatespeech → hate speech
Offensive  → offensive language
Normal     → neither
```

---

# 12. Optional Model E — Detoxify

Detoxify may be included as an **auxiliary comparison**, but it must not be presented as a direct apples-to-apples competitor.

Detoxify produces multiple toxicity-related scores such as:

```text
toxicity
severe_toxicity
obscene
threat
insult
identity_attack
```

Its models were trained using the Jigsaw toxicity datasets rather than the Davidson three-class dataset.

Therefore:

```text
Davidson 3-class classification
≠
Detoxify multi-label toxicity classification
```

For Detoxify, the report should use a separate experiment such as:

```text
hate speech vs non-hate
```

and clearly label this as a **secondary analysis**.

It must not be mixed directly with the three-class macro-F1 table.

---

# 13. Standardized Evaluation

Every comparable model must process exactly the same test rows.

Input:

```text
tweet text
```

Output:

```text
predicted class
```

The evaluator should create one unified result table:

```text
id
text
gold_label

majority_prediction

tfidf_prediction

laya_prediction
laya_hate_probability
laya_offensive_probability
laya_neither_probability

hatexplain_prediction

detoxify_toxicity      # optional
detoxify_identity_attack # optional
```

This file becomes the central artifact of the experiment.

---

# 14. Metrics

## Primary metrics

### Accuracy

Overall percentage of correctly classified examples.

### Macro F1

Primary multiclass comparison metric.

Macro F1 is important because it gives equal weight to each class.

### Per-class precision

Measure how reliable each predicted class is.

### Per-class recall

Measure how many examples belonging to each class are detected.

### Per-class F1

Combined precision/recall metric.

---

# 15. Hate-Speech Specific Metrics

Because hate speech is the main research target, the report must explicitly expose:

```text
Hate precision
Hate recall
Hate F1
```

Example:

```text
Hate recall = correctly detected hate tweets /
             all actual hate tweets
```

This must be displayed independently from overall accuracy.

The experiment must also report:

```text
False negatives for hate speech
False positives for hate speech
```

---

# 16. Confusion Matrix

Generate one confusion matrix per three-class model.

Example:

```text
                 Predicted
             Hate  Off.  Neither
Actual Hate
Actual Off.
Actual Neither
```

This is especially important for identifying the difference between:

```text
hate speech ↔ offensive language
```

because a system may appear strong overall while systematically confusing those two categories.

---

# 17. Latency Benchmark

Because this is a local-model experiment, inference performance should also be measured.

Collect:

```text
total inference time
average latency
p50 latency
p95 latency
throughput
```

Latency measurement must exclude:

* model download
* model loading
* dataset download

Those should be reported separately.

Recommended phases:

```text
Cold start
Model load
Warm inference
```

---

# 18. Hardware/Runtime Information

The experiment must record:

```text
OS
Python version
PyTorch version
Transformers version
Laya version / commit
Model checkpoint
CPU
RAM
GPU
GPU memory
CUDA/device configuration
```

This allows another person to understand whether performance numbers are reproducible.

---

# 19. Reproducibility

The repository must contain a single command capable of reproducing the evaluation.

Target:

```bash
python run_benchmark.py
```

The script should:

1. Validate environment.
2. Load/download dataset.
3. Verify dataset hash.
4. Create/load fixed split.
5. Load models.
6. Execute predictions.
7. Calculate metrics.
8. Generate charts.
9. Generate report.
10. Save results.

---

# 20. Output Artifacts

The experiment must generate:

```text
results/
│
├── dataset_metadata.json
├── experiment_config.json
├── split.json
│
├── predictions.csv
├── metrics.json
│
├── confusion_matrix_majority.png
├── confusion_matrix_tfidf.png
├── confusion_matrix_laya.png
├── confusion_matrix_hatexplain.png
│
├── model_comparison.csv
├── error_analysis.csv
│
└── report.md
```

Optional:

```text
report.html
report.pdf
```

---

# 21. Main Comparison Table

The final report must contain a table similar to:

| Model           | Accuracy | Macro F1 | Hate Precision | Hate Recall | Hate F1 | p50 Latency |
| --------------- | -------: | -------: | -------------: | ----------: | ------: | ----------: |
| Majority        |          |          |                |             |         |             |
| TF-IDF + LR     |          |          |                |             |         |             |
| Laya            |          |          |                |             |         |             |
| HateXplain BERT |          |          |                |             |         |             |

No single metric should be used as the sole basis for interpretation.

---

# 22. Error Analysis

A small sample of incorrectly classified examples must be exported.

Recommended:

```text
20–50 examples per important error category
```

Examples of categories:

```text
Gold: hate speech
Predicted: offensive language

Gold: hate speech
Predicted: neither

Gold: offensive language
Predicted: hate speech

Gold: neither
Predicted: hate speech
```

For each example:

```text
text
gold label
predicted label
confidence/probability if available
```

The report should investigate recurring patterns rather than simply listing errors.

---

# 23. Optional Confidence Analysis

For models that provide probabilities, calculate:

```text
confidence distribution
correct vs incorrect confidence
```

Optionally include:

* calibration curve
* Brier score
* Expected Calibration Error (ECE)

This should remain secondary to the core benchmark.

---

# 24. Laya Fine-Tuning — Phase 2

Fine-tuning Laya should **not** be part of the mandatory MVP.

It can be added after the base comparison.

Purpose:

```text
Laya base
     vs
Laya fine-tuned
```

using exactly the same train/test protocol.

This is particularly interesting because the published experiment showed that fine-tuning improved overall accuracy and macro F1 but reduced hate-speech recall under that specific setup.

If implemented, the report must clearly distinguish:

```text
zero-shot/base model
```

from:

```text
supervised fine-tuned model
```

because they are fundamentally different experimental conditions.

---

# 25. Proposed Repository Structure

```text
laya-hate-speech-benchmark/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── dataset.py
│   ├── metrics.py
│   ├── evaluation.py
│   ├── models/
│   │   ├── majority.py
│   │   ├── tfidf.py
│   │   ├── laya.py
│   │   └── hatexplain.py
│   └── reporting.py
│
├── scripts/
│   └── run_benchmark.py
│
├── results/
│
└── report/
    └── benchmark_report.md
```

---

# 26. Suggested Execution Flow

```text
                 Davidson Dataset
                         │
                         ▼
                 Fixed Train/Test Split
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      TF-IDF + LR       Laya       HateXplain
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Unified Predictions
                         │
                         ▼
                      Metrics
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
      Accuracy        Macro F1     Hate Recall
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Error Analysis
                         │
                         ▼
                    Final Report
```

---

# 27. Acceptance Criteria

The PoC is considered complete when:

### AC1

The Davidson dataset can be downloaded automatically and verified.

### AC2

A deterministic train/test split is generated and persisted.

### AC3

Laya can process the complete test set locally.

### AC4

TF-IDF + Logistic Regression produces predictions on the same test set.

### AC5

HateXplain BERT produces predictions on the same test set.

### AC6

All three models plus the majority baseline are evaluated with:

```text
accuracy
macro F1
per-class precision
per-class recall
per-class F1
```

### AC7

Hate-speech precision/recall/F1 are explicitly reported.

### AC8

A confusion matrix is produced for every comparable model.

### AC9

Latency measurements are recorded.

### AC10

All predictions are stored in one CSV/Parquet file.

### AC11

A Markdown report is generated automatically.

### AC12

Running the benchmark twice with the same configuration produces the same split and equivalent metrics within documented numerical tolerances.

---

# 28. Definition of Success

The goal of this project is **not** to prove beforehand that Laya is better or worse than another model.

The goal is to produce enough evidence to answer:

```text
1. How does Laya perform on this dataset?

2. How does it compare with a simple classical baseline?

3. How does it compare with a specialized
   hate-speech classifier?

4. Where does each approach make mistakes?

5. What is the tradeoff between overall accuracy,
   hate-speech recall, precision and local inference cost?

6. Does fine-tuning materially change the behavior
   of Laya?
```

The conclusion must be based on the measured results rather than a single benchmark number.

---

# 29. Recommended MVP

To keep the project small, the first implementation should contain only:

```text
Dataset:
    Davidson

Models:
    1. Majority baseline
    2. TF-IDF + Logistic Regression
    3. Laya base
    4. HateXplain BERT

Metrics:
    Accuracy
    Macro F1
    Precision
    Recall
    Per-class F1
    Hate recall
    Hate precision

Performance:
    p50 latency
    p95 latency

Artifacts:
    predictions.csv
    metrics.json
    confusion matrices
    benchmark_report.md
```

Do **not** implement fine-tuning, Detoxify, web UI, APIs or other datasets until this MVP produces a clean reproducible result.

---

# 30. Phase 2 Extensions

After the MVP is working:

```text
Phase 2A
Laya fine-tuning

Phase 2B
Detoxify auxiliary comparison

Phase 2C
Second hate-speech dataset

Phase 2D
Calibration analysis

Phase 2E
GPU vs CPU benchmark

Phase 2F
Cross-dataset generalization
```

The most valuable Phase 2 experiment would be testing whether a model performs consistently on a **different hate-speech dataset**, because a strong result on a single dataset does not establish generalization.

---

# 31. Final Deliverable

The final project should produce one document:

**`benchmark_report.md`**

with the following structure:

```text
1. Executive Summary
2. Research Question
3. Dataset
4. Experimental Setup
5. Models
6. Evaluation Methodology
7. Overall Results
8. Hate-Speech Results
9. Confusion Matrices
10. Latency Results
11. Error Analysis
12. Laya Analysis
13. Limitations
14. Conclusions
15. Reproduction Instructions
16. References
```

The report must distinguish clearly between:

```text
measured result
```

and:

```text
interpretation
```

and must document the exact model versions, dataset version, split, hardware and software environment used.

---

## 32. Initial Technical Decision

For the first implementation, use:

```text
Python
PyTorch
Transformers
scikit-learn
pandas
Laya
Hugging Face Datasets
matplotlib
```

The project should favor a simple Python CLI over notebooks as the primary execution mechanism.

A notebook can optionally be added later for exploratory analysis.

---

## 33. Reference Sources

Davidson et al., *Automated Hate Speech Detection and the Problem of Offensive Language*, ICWSM 2017.

Davidson hate-speech/offensive-language dataset, 24,783 English tweets and three classes.

Independent Laya benchmark and fine-tuning experiment using the Davidson dataset.

Laya repository and decision-engine documentation.

HateXplain BERT three-class classifier.

Detoxify documentation and model taxonomy.
