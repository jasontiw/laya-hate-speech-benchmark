#!/usr/bin/env python
"""Phase 2A step 1 — turn the Davidson split into Laya fine-tuning JSONL.

The fine-tune must see *exactly* the question the harness evaluates, otherwise the
comparison is between two different tasks. This script therefore reads the question
wording straight out of ``config.yaml`` (the L1 variant by default) and reuses
``src.dataset`` to rebuild the same deterministic, duplicate-grouped split that
``run_benchmark.py`` uses. No test row is ever read here.

Each output line is one tweet::

    {"id": 1, "state": "<raw tweet>",
     "questions": {"label": {"type": "choice", "instructions": ..., "criteria": {...}}},
     "gold": {"label": {"probabilities": {"hate speech": 0.0, ...}, "label": "offensive language"}}}

Gold targets are **one-hot**, because Davidson ships hard labels. The reference
notebook trains on teacher soft probabilities; that is a documented difference, not
a hidden one.

Usage:
    python scripts/build_laya_finetune_data.py
    python scripts/build_laya_finetune_data.py --variant laya --limit 200 --out-dir data/processed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CANONICAL_LABELS, Config, load_config  # noqa: E402
from src.dataset import build_split, load_raw  # noqa: E402


def _find_variant(cfg: Config, key: str):
    for variant in cfg.laya.variants:
        if variant.key == key:
            return variant
    raise SystemExit(
        "variant %r not found in config.yaml; available: %s"
        % (key, ", ".join(v.key for v in cfg.laya.variants))
    )


def _question_for(variant) -> Dict[str, Any]:
    """The same dict ``LayaClassifier._build_question`` sends for a choice variant."""
    return {
        "label": {
            "type": variant.question_type,
            "instructions": variant.instructions,
            "criteria": dict(variant.criteria),
        }
    }


def build_cases(frame: pd.DataFrame, variant) -> List[Dict[str, Any]]:
    questions = _question_for(variant)
    criteria = list(variant.criteria.keys())
    if criteria != CANONICAL_LABELS:
        raise SystemExit(
            "variant %r criteria %s are not the canonical classes %s; the benchmark "
            "comparison would not be apples-to-apples" % (variant.key, criteria, CANONICAL_LABELS)
        )
    cases: List[Dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        gold_label = str(row.gold_label)
        probabilities = {label: (1.0 if label == gold_label else 0.0) for label in criteria}
        cases.append({
            "id": int(row.id),
            "state": str(row.text),
            "questions": json.loads(json.dumps(questions)),  # one owned copy per case
            "gold": {"label": {"probabilities": probabilities, "label": gold_label}},
        })
    return cases


def _write_jsonl(path: Path, cases: List[Dict[str, Any]]) -> Tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as fh:
        for case in cases:
            line = json.dumps(case, ensure_ascii=False)
            digest.update(line.encode("utf-8"))
            fh.write(line + "\n")
    return digest.hexdigest(), len(cases)


def _distribution(cases: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {label: 0 for label in CANONICAL_LABELS}
    for case in cases:
        counts[case["gold"]["label"]["label"]] += 1
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    parser.add_argument("--variant", default="laya",
                        help="Laya variant key whose question wording the fine-tune reuses (default: laya = L1)")
    parser.add_argument("--out-dir", default="data/processed", help="where to write the JSONL files")
    parser.add_argument("--limit", type=int, default=None, help="DEBUG: cap rows per split")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    variant = _find_variant(cfg, args.variant)

    raw_df, _ = load_raw(cfg)
    frame, split_info = build_split(raw_df, cfg)

    train_df = frame[frame["split"] == "train"].reset_index(drop=True)
    validation_df = frame[frame["split"] == "validation"].reset_index(drop=True)
    if args.limit:
        train_df = train_df.head(args.limit).reset_index(drop=True)
        validation_df = validation_df.head(args.limit).reset_index(drop=True)

    out_dir = PROJECT_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    train_cases = build_cases(train_df, variant)
    validation_cases = build_cases(validation_df, variant)

    train_path = out_dir / "laya_finetune_train.jsonl"
    validation_path = out_dir / "laya_finetune_validation.jsonl"
    train_sha, n_train = _write_jsonl(train_path, train_cases)
    validation_sha, n_validation = _write_jsonl(validation_path, validation_cases)

    print("variant      : %s (%s)" % (variant.key, variant.label))
    print("question     : %s" % variant.instructions)
    print("train        : %d rows -> %s" % (n_train, train_path))
    print("               sha256=%s" % train_sha)
    print("               classes=%s" % _distribution(train_cases))
    print("validation   : %d rows -> %s" % (n_validation, validation_path))
    print("               sha256=%s" % validation_sha)
    print("               classes=%s" % _distribution(validation_cases))
    print("test_rows_used: 0 (test_id_sha256=%s)" % str(split_info["test_id_sha256"])[:12])
    print("\nnext: python scripts/finetune_laya.py --data %s --output-dir artifacts/laya-finetuned-l1"
          % train_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
