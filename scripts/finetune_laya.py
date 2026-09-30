#!/usr/bin/env python
"""Phase 2A step 2 — fine-tune a Laya checkpoint on the Davidson train split.

This is a single-device port of Laya's own fine-tuning loop (upstream
``research/scripts/finetune_single_device.py`` / the 2xT4 Kaggle notebook). The
training objective is unchanged: RLCD — a GRPO-style policy gradient against a
strictly proper scoring rule, plus a soft cross-entropy term on the gold
distribution. Only the plumbing differs:

* the data is our JSONL (``scripts/build_laya_finetune_data.py``), not the
  ``LocalLLaMA/typed-decisions`` dataset,
* gold targets are one-hot, because Davidson has hard labels,
* no temperature is fitted here. Temperature is fitted on the benchmark's
  validation split by the harness, exactly as it is for the zero-shot variants,
  so both regimes are calibrated by the same procedure. Pass ``--fit-temperature``
  to reproduce the upstream in-script fit instead.

It runs on one GPU (or CPU, slowly) and writes a directory that ``laya.load``
accepts directly::

    scripts/finetune_laya.py --data data/processed/laya_finetune_train.jsonl \
        --output-dir artifacts/laya-finetuned-l1

    # smoke test: ~200 rows, 1 epoch, measures the real step time
    scripts/finetune_laya.py --data ... --limit 200 --epochs 1 --output-dir /tmp/smoke

Nothing here reads the test split.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import (  # noqa: E402
    QTYPES,
    build_model,
    build_sequence,
    proper_reward,
    render_options,
)

HATE_INDEX = 0  # canonical class order: hate speech, offensive language, neither


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_training_item(tok, cfg: Dict[str, Any], state: str, q: Dict[str, Any],
                        gold_q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Tokenize one (state, question) pair into a training item.

    Returns ``None`` when the tokenizer could not give every option its own marker,
    which is the same condition Laya itself checks before it will answer a question.
    """
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [float(gold_q["probabilities"].get(k, 0.0)) for k in keys]
    elif t == "noul":
        target = [
            float(gold_q["probabilities"].get("false", 0.5)),
            float(gold_q["probabilities"].get("true", 0.5)),
        ]
    else:
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [float(gold_q["probabilities"].get(str(i), 0.0)) for i in range(n_levels)]
    total = sum(target)
    target = [v / total for v in target] if total > 0 else [1.0 / len(target)] * len(target)

    seq, markers = build_sequence(
        tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
        cfg["max_len"], cfg["head_max_len"],
    )
    if len(markers) != len(render_options({"t": t, "crit": crit})):
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[t],
        "target": target,
        "label": target.index(max(target)),
    }


def preprocess(tok, cfg: Dict[str, Any], data_path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for qid, q in row["questions"].items():
                if qid not in row["gold"]:
                    continue
                item = build_training_item(tok, cfg, row["state"], q, row["gold"][qid])
                if item is not None:
                    items.append(item)
    return items


def collate(items: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    n, length = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def forward(model, batch: Dict[str, torch.Tensor], device, amp: str):
    if amp in ("fp16", "bf16"):
        dtype = torch.float16 if amp == "fp16" else torch.bfloat16
        with torch.autocast("cuda", dtype=dtype):
            return model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
    return model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["marker_pos"].to(device),
        batch["marker_mask"].to(device),
        batch["qtype"].to(device),
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _hate_metrics(gold: List[int], pred: List[int]) -> Dict[str, Any]:
    tp = sum(1 for g, p in zip(gold, pred) if g == HATE_INDEX and p == HATE_INDEX)
    fp = sum(1 for g, p in zip(gold, pred) if g != HATE_INDEX and p == HATE_INDEX)
    fn = sum(1 for g, p in zip(gold, pred) if g == HATE_INDEX and p != HATE_INDEX)
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / max(1, len(gold))
    return {
        "rows": len(gold),
        "accuracy": accuracy,
        "hate_precision": tp / (tp + fp) if (tp + fp) else None,
        "hate_recall": tp / (tp + fn) if (tp + fn) else None,
        "hate_tp": tp,
        "hate_fp": fp,
        "hate_fn": fn,
    }


def evaluate(model, items: List[Dict[str, Any]], tok, device, amp: str, batch_size: int = 16) -> Dict[str, Any]:
    """Argmax accuracy and hate-vs-rest numbers, the same shape the harness reports."""
    if not items:
        return {}
    was_training = model.training
    model.eval()
    gold: List[int] = []
    pred: List[int] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            batch = collate(chunk, tok.pad_token_id)
            logits, _ = forward(model, batch, device, amp)
            logits = logits.float().cpu()
            for row, item in enumerate(chunk):
                k = len(item["markers"])
                pred.append(int(logits[row, :k].argmax()))
                gold.append(int(item["label"]))
    if was_training:
        model.train()
    return _hate_metrics(gold, pred)


# --------------------------------------------------------------------------- #
# Temperature (opt-in: upstream fits this in-script)
# --------------------------------------------------------------------------- #
def fit_one_temperature(selection: List[Tuple[Any, Any]]) -> float:
    if len(selection) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in selection)
    Z = torch.full((len(selection), kmax), -1e4)
    T = torch.zeros((len(selection), kmax))
    for i, (z, t) in enumerate(selection):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def fit_temperatures(model, items: List[Dict[str, Any]], tok, device, amp: str) -> List[float]:
    """Fit one temperature per question type on `items`; returns [choice, score, noul]."""
    model.eval()
    predictions: List[Tuple[int, Any, Any]] = []
    with torch.no_grad():
        for start in range(0, len(items), 16):
            chunk = items[start:start + 16]
            batch = collate(chunk, tok.pad_token_id)
            logits, _ = forward(model, batch, device, amp)
            values = logits.float().cpu().numpy()
            for row, item in enumerate(chunk):
                predictions.append((item["qtype"], values[row, : len(item["markers"])], item["target"]))
    fitted = [1.2, 1.2, 1.2]
    for qtype in range(3):
        selection = [(z, t) for kind, z, t in predictions if kind == qtype]
        if selection:
            fitted[qtype] = fit_one_temperature(selection)
    return fitted


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _resolve_model_dir(model_dir: Optional[str]) -> str:
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download

    return snapshot_download(
        "convaiinnovations/laya",
        allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
    )


def _git_commit() -> Optional[str]:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Laya checkpoint on the Davidson train split (single device).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="training JSONL from scripts/build_laya_finetune_data.py")
    parser.add_argument("--eval-data", default=None, help="optional JSONL to score before and after training")
    parser.add_argument("--model-dir", default=None,
                        help="Laya checkpoint directory (default: download the English checkpoint root)")
    parser.add_argument("--output-dir", default="artifacts/laya-finetuned-l1")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=8, help="sequences per forward pass")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="micro-batches per optimizer step (the notebook uses 4 across 2 GPUs)")
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr-head", type=float, default=1.0e-4)
    parser.add_argument("--group-size", type=int, default=4, help="GRPO baseline samples")
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--w-sph", type=float, default=0.75, help="spherical term weight of the reward")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"])
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_true", default=None,
                        help="force encoder gradient checkpointing on (default: on for CUDA)")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false",
                        help="disable gradient checkpointing (needs more VRAM, runs faster)")
    parser.add_argument("--limit", type=int, default=None, help="DEBUG: cap training items")
    parser.add_argument("--max-steps", type=int, default=None, help="DEBUG: stop after N optimizer steps")
    parser.add_argument("--log-every", type=int, default=25, help="print progress every N optimizer steps")
    parser.add_argument("--fit-temperature", action="store_true",
                        help="also fit temperatures on a 10%% slice of the training items (upstream behaviour)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    amp = args.amp if use_cuda else "none"
    gradient_checkpointing = args.gradient_checkpointing
    if gradient_checkpointing is None:
        gradient_checkpointing = use_cuda

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model_dir = _resolve_model_dir(args.model_dir)
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)

    # Train at the same token budget the harness evaluates with, so the fine-tuned
    # model sees exactly the sequence shape inference will give it.
    cfg["max_len"] = int(cfg.get("max_len", 512))
    cfg["head_max_len"] = int(cfg.get("head_max_len", 192))
    cfg["gradient_checkpointing"] = bool(gradient_checkpointing)
    cfg["max_tokens_per_batch"] = 4096

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    if gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True
    model.to(device)
    model.train()

    print("device            : %s" % device)
    print("amp               : %s" % amp)
    print("grad checkpointing: %s" % gradient_checkpointing)
    print("model dir         : %s" % model_dir)
    print("max_len/head      : %d/%d" % (cfg["max_len"], cfg["head_max_len"]))

    items = preprocess(tok, cfg, args.data)
    if not items:
        raise SystemExit("No training items produced from --data.")
    if args.limit:
        random.seed(args.seed)
        random.shuffle(items)
        items = items[: args.limit]

    # A calibration slice is carved out ONLY when --fit-temperature asks for the upstream
    # in-script fit. With the benchmark's validation split doing the fitting, nothing needs
    # holding out: the model trains on every train row, which is also what the PRD protocol
    # specifies. The validation split is disjoint from train, so there is no leakage either way.
    n_calib = min(400, len(items) // 10) if args.fit_temperature else 0
    if n_calib:
        order = list(range(len(items)))
        random.Random(args.seed).shuffle(order)
        calib_items = [items[i] for i in sorted(order[:n_calib])]
        train_items = [items[i] for i in sorted(order[n_calib:])]
    else:
        calib_items = []
        train_items = list(items)

    if calib_items:
        print("train items       : %d (%d held out for in-script calibration)" % (len(train_items), len(calib_items)))
    else:
        print("train items       : %d (no in-script calibration slice; the harness fits temperatures "
              "on validation)" % len(train_items))

    eval_items = preprocess(tok, cfg, args.eval_data) if args.eval_data else []
    eval_history: List[Dict[str, Any]] = []
    baseline_metrics: Optional[Dict[str, Any]] = None
    if eval_items:
        random.Random(args.seed).shuffle(eval_items)
        eval_items = eval_items[: args.limit] if args.limit else eval_items
        print("eval items        : %d" % len(eval_items))
        baseline_metrics = evaluate(model, eval_items, tok, device, amp)
        baseline_metrics["epoch"] = 0
        baseline_metrics["step"] = 0
        eval_history.append(baseline_metrics)
        print("before training   : %s" % json.dumps(baseline_metrics))

    encoder_params = [p for name, p in model.named_parameters() if "encoder." in name]
    head_params = [p for name, p in model.named_parameters() if "encoder." not in name]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )
    micro_batch = args.micro_batch
    grad_accum = max(1, args.grad_accum)
    steps_per_epoch = max(1, (len(train_items) + micro_batch * grad_accum - 1) // (micro_batch * grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16")) if use_cuda else None

    global_step = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        random.seed(args.seed + epoch)
        random.shuffle(train_items)
        progress = epoch / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pending = 0

        for b_idx in range(0, len(train_items), micro_batch):
            chunk = train_items[b_idx:b_idx + micro_batch]
            if not chunk:
                continue
            batch = collate(chunk, tok.pad_token_id)
            logits, _act = forward(model, batch, device, amp)
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. Sample G noisy logit distributions with a zero-mean projection.
            eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Score them with the proper scoring rule; GRPO advantage.
            with torch.no_grad():
                reward = proper_reward(
                    q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                    w_sph=args.w_sph, w_rps=1.0,
                )
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)

            # 3. Policy gradient on the sampled distributions + soft cross-entropy target.
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(advantage * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / grad_accum

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            pending += 1

            last_in_epoch = (b_idx + micro_batch) >= len(train_items)
            if pending % grad_accum == 0 or last_in_epoch:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.log_every and global_step % args.log_every == 0:
                    elapsed = time.perf_counter() - started
                    rate = global_step / elapsed if elapsed else 0.0
                    print(
                        "  epoch %d/%d | step %d/%d | loss %.4f | reward %.3f | lr %.2e | %.2f step/s"
                        % (epoch + 1, args.epochs, global_step, steps_per_epoch * args.epochs,
                           loss.item() * grad_accum, float(reward.mean()), scheduler.get_last_lr()[0], rate),
                        flush=True,
                    )
                if args.max_steps and global_step >= args.max_steps:
                    break

            epoch_loss += loss.item() * grad_accum
            n_batches += 1

        print(
            "epoch %d/%d | avg loss %.4f | %.1fs elapsed"
            % (epoch + 1, args.epochs, epoch_loss / max(1, n_batches), time.perf_counter() - started),
            flush=True,
        )
        if eval_items:
            metrics = evaluate(model, eval_items, tok, device, amp)
            metrics["epoch"] = epoch + 1
            metrics["step"] = global_step
            eval_history.append(metrics)
            print("epoch %d/%d | validation: %s" % (epoch + 1, args.epochs, json.dumps(metrics)), flush=True)
        if args.max_steps and global_step >= args.max_steps:
            print("stop: --max-steps %d reached" % args.max_steps)
            break

    if eval_items:
        print("after training    : %s" % json.dumps(evaluate(model, eval_items, tok, device, amp)))

    fitted = None
    if args.fit_temperature and calib_items:
        fitted = fit_temperatures(model, calib_items, tok, device, amp)
        print("fitted temperatures (choice, score, noul): %s" % [round(t, 3) for t in fitted])

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    save_file(
        {key: value.half().contiguous().cpu() for key, value in model.state_dict().items()},
        os.path.join(args.output_dir, "model.safetensors"),
    )
    model.encoder.config.save_pretrained(os.path.join(args.output_dir, "encoder"))
    tok.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    cfg["fine_tuned"] = True
    if fitted is not None:
        cfg["temperature"] = fitted
        # A per-type fit supersedes the inherited per-bucket overrides, which would hide it.
        cfg.pop("temperature_by_options", None)
    cfg["finetune"] = {
        "base_checkpoint": model_dir,
        "data": os.path.abspath(args.data),
        "train_items": len(train_items),
        "calibration_items_held_out": len(calib_items),
        "epochs": args.epochs,
        "optimizer_steps": global_step,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "group_size": args.group_size,
        "lr_encoder": args.lr_encoder,
        "lr_head": args.lr_head,
        "sigma_start": args.sigma_start,
        "sigma_end": args.sigma_end,
        "amp": amp,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "temperature_fitted_here": fitted is not None,
        "eval_on": os.path.abspath(args.eval_data) if args.eval_data else None,
        "eval_history": eval_history,
        "seed": args.seed,
        "device": str(device),
        "seconds": round(time.perf_counter() - started, 1),
        "git_commit": _git_commit(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    print("saved checkpoint  : %s" % args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
