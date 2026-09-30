# Phase 2A — Laya fine-tuning

PRD section 24 defers fine-tuning out of the mandatory MVP; section 30 lists it as
Phase 2A. This page documents how it is run here, what it does and does not establish,
and how to reproduce it.

The benchmark itself never tunes Laya: the zero-shot rows are the published
checkpoints. Everything below produces a **local checkpoint artifact** which the
harness then evaluates exactly like any other model row.

---

## Why this is an experiment, not a chore

The conclusion of the zero-shot benchmark had one large hole: *without fine-tuning we
do not know Laya's ceiling on this dataset*. That is an open question, and open
questions are not falsifiable. The reference experiment in the PRD turns it into one:

| Model (reference protocol) | Accuracy | Macro F1 | Hate recall |
| --- | ---: | ---: | ---: |
| Original Laya | 22.10% | 0.2228 | 77.97% |
| Fine-tuned Laya | 82.75% | 0.6521 | 41.43% |

Fine-tuning moved that protocol toward **precision** and away from **recall**. Our
zero-shot L1 is oriented the other way: **68.07% accuracy / 67.25% hate recall**.

So the prediction, registered before the run:

> Fine-tuning L1 on the Davidson train split should **raise accuracy and macro F1** and
> **lower hate recall** relative to L1 zero-shot on the same test rows.

The registered prediction is also quoted in the report when the fine-tuned row is
present, so it is judged against the measurement rather than remembered afterwards.
It is a direction, not a target: if the measurement disagrees, the report says so.

---

## Protocol

### 1. Build the data from the benchmark's own split

```powershell
.\.venv\Scripts\python.exe scripts\build_laya_finetune_data.py
```

This reads the L1 question wording straight out of `config.yaml` and rebuilds the same
deterministic, duplicate-grouped split that `run_benchmark.py` uses, so the fine-tune is
trained on **exactly** the question the harness later evaluates. It writes
`data/processed/laya_finetune_train.jsonl` (16,130 rows) and
`data/processed/laya_finetune_validation.jsonl` (3,720 rows). **No test row is read.**

Gold targets are **one-hot** (`{"hate speech": 1.0, ...}`), because Davidson ships hard
labels. The reference notebook trains on teacher *soft* probabilities from a frontier
model; that difference is stated in the report rather than hidden.

### 2. Fine-tune

```powershell
.\.venv\Scripts\python.exe scripts\finetune_laya.py `
    --data data\processed\laya_finetune_train.jsonl `
    --eval-data data\processed\laya_finetune_validation.jsonl `
    --output-dir artifacts\laya-finetuned-l1
```

`scripts/finetune_laya.py` is a single-device port of Laya's own loop (upstream
`research/scripts/finetune_single_device.py` and the 2xT4 Kaggle notebook). The
objective is unchanged: **RLCD** — a GRPO-style policy gradient against a strictly
proper scoring rule (log score + spherical score + ranked probability score), plus a
soft cross-entropy term on the gold distribution.

Defaults follow the notebook:

| Setting | Value |
| --- | --- |
| Epochs | 4 |
| Micro-batch | 8 sequences per forward pass |
| Gradient accumulation | 8 → effective batch 64 |
| Optimizer | AdamW, `lr` 2.5e-5 (encoder) / 1.0e-4 (head), weight decay 0.01 |
| Schedule | Cosine annealing to 1e-6 |
| Group size (GRPO) | 4 |
| Exploration noise | sigma 0.4 → 0.1 across epochs |
| Precision | fp16 autocast + `GradScaler` |
| Gradient checkpointing | on (CUDA) |

Validation accuracy / hate recall is evaluated before training and after every epoch,
and stored in the checkpoint's `rl_agent_config.json` under `finetune.eval_history`, so
the trajectory is part of the artifact rather than a number in a log file.

> **Fidelity check.** The pre-training evaluation of the untouched checkpoint on the
> validation split gives **67.82% accuracy / 65.73% hate recall**. The harness reports L1
> zero-shot on the *test* split as **68.07% / 67.25%**. Two different splits, two
> independent evaluation code paths, agreeing to about a point — the fine-tune starts from
> the same model the benchmark measured, not from a differently-prepared one.

The trajectory is not monotone in either direction, and that is the interesting part. On
the validation split:

| Epoch | Accuracy | Hate precision | Hate recall |
| ---: | ---: | ---: | ---: |
| 0 — zero-shot | 67.82% | 0.175 | 0.657 |
| 1 | 90.89% | 0.386 | 0.080 |
| 2 | 91.13% | 0.492 | 0.141 |
| 3 | **91.42%** | 0.529 | 0.258 |
| 4 | 91.05% | 0.514 | **0.333** |

The first epoch over-commits to precision and nearly abandons the minority class; epochs
2–4 buy hate recall back while accuracy holds. So the fine-tune does not simply learn the
majority prior — it trades against it and then partially reverses. Reading only the last
epoch, or only the first, would misdescribe what happened, which is why the full
trajectory is an artifact (and why the test numbers, not these, are the headline).

Three deltas from the reference notebook, all deliberate:

1. **One device instead of 2×T4 DDP.** Same math; the notebook's *effective* batch of 64
   is preserved with `--grad-accum 8`.
2. **No temperature fitted inside the training script** (pass `--fit-temperature` to get
   it). Temperatures are fitted on the benchmark's validation split by the harness, the
   same code path used for the zero-shot variants, so both regimes are calibrated
   identically and the comparison is not confounded by two different calibrators. Because
   nothing needs holding out when the in-script fit is off, the model trains on **all
   16,130 train rows**; the validation split it does not train on is disjoint.
3. **Hard labels**, as described above.

### 3. Evaluate it on the untouched test split

```powershell
.\.venv\Scripts\python.exe run_benchmark.py --include-finetuned
```

`models.laya_finetuned` is **disabled by default**: the checkpoint is a ~0.8 GB derived
artifact, so a fresh clone must still reproduce the zero-shot benchmark with one command.
`--include-finetuned` flips it on and the row appears in every table with training regime
*Supervised fine-tuned (RLCD, fit here)*.

---

## Cost, measured

On the machine recorded in `results/experiment_config.json` (RTX 4060 Ti 16 GB, 17.2 GB
VRAM), measured during this project:

| Shape | Measured rate | Sequences/s |
| --- | --- | --- |
| micro-batch 8, `--grad-accum 1` (smoke run) | **~3.0 optimizer steps/s** | ~24 |
| micro-batch 8, `--grad-accum 8` (the real run) | **~0.63 optimizer steps/s** | ~40 |

The real 4-epoch run is 1,012 optimizer steps, so **~27 minutes**, plus one validation
pass per epoch. Larger accumulation does not cost wall time per gradient; it amortises the
optimizer step over 64 sequences and raises throughput. Measured end to end: **1,650 s
(27.5 min)** for 1,012 steps and 5 validation passes, including checkpoint save.

This is the same order as the reference: the shipped English checkpoint's own
`rl_agent_config.json` records 7,313 updates in 1.96 h on a single device.

Run the smoke first. It validates the pipeline end to end and gives the real step rate
before committing the full run:

```powershell
.\.venv\Scripts\python.exe scripts\finetune_laya.py `
    --data data\processed\laya_finetune_train.jsonl `
    --limit 200 --epochs 1 --max-steps 25 --output-dir artifacts\smoke
```

---

## What this does and does not establish

**Does.** Whether fine-tuning moves Laya on *this* dataset, *this* split and *this*
question, measured against the zero-shot row on identical test rows with identical
metrics — the falsifiable contrast PRD section 24 asks for.

**Does not.**

- **Choose a hyperparameter set.** One seed, one effective batch, one epoch count.
  Nothing was selected on the test set, but nothing was tuned either.
- **Establish a ceiling.** A single run bounds Laya's performance from below, not from
  above. Soft teacher targets, longer training, or a different head learning rate could
  all move it.
- **Compare architectures.** As everywhere in this benchmark, the fine-tuned row is a
  *training regime*, not evidence about the architecture.
- **Generalize.** Same dataset, same language, same split seed. That is Phase 2C/2F.

---

## Implementation map

| File | Role |
| --- | --- |
| `scripts/build_laya_finetune_data.py` | split → JSONL, reusing the L1 question verbatim |
| `scripts/finetune_laya.py` | single-device RLCD fine-tune, writes a `laya.load`-able checkpoint |
| `src/config.py` | `models.laya_finetuned`, `TRAINING_REGIME["laya_finetuned"]` |
| `src/models/__init__.py` | `build_finetuned_laya`: mounts the checkpoint as a model row |
| `src/reporting.py` | the zero-shot / fine-tuned contrast section |
| `run_benchmark.py` | `--include-finetuned` |
