"""Model registry for the benchmark.

The registry is built from the configuration rather than hardcoded, because a run
compares *variants* as well as models: TF-IDF with and without balanced class
weights, and several Laya prompt formulations. Each variant gets its own key, so
it becomes its own row in every artifact.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import PROJECT_ROOT, Config, LayaConfig
from .base import Classifier, ModelResult
from .hatexplain import HateXplainClassifier
from .laya_model import LayaClassifier
from .majority import MajorityClassifier
from .tfidf import TfidfLogisticClassifier

__all__ = [
    "Classifier",
    "ModelResult",
    "MajorityClassifier",
    "TfidfLogisticClassifier",
    "LayaClassifier",
    "HateXplainClassifier",
    "available_model_keys",
    "build_models",
]

FINETUNED_KEY = "laya_finetuned"


def available_model_keys(cfg: Config) -> List[str]:
    """The model keys this configuration would run, in reporting order."""
    keys: List[str] = []
    if cfg.majority.enabled:
        keys.append("majority")
    if cfg.tfidf.enabled:
        keys.extend(variant.key for variant in cfg.tfidf.variants)
    if cfg.laya.enabled:
        keys.extend(variant.key for variant in cfg.laya.variants)
    if cfg.laya_finetuned.enabled:
        keys.append(FINETUNED_KEY)
    if cfg.hatexplain.enabled:
        keys.append("hatexplain")
    return keys


def build_finetuned_laya(cfg: Config) -> LayaClassifier:
    """A Laya checkpoint written by ``scripts/finetune_laya.py``, as a benchmark model.

    It reuses the question wording of the zero-shot variant it was trained on, so the
    fine-tuned row answers exactly the same question — otherwise the two rows would not
    be comparable.
    """
    source = next((v for v in cfg.laya.variants if v.key == cfg.laya_finetuned.source_variant), None)
    if source is None:
        raise RuntimeError(
            "laya_finetuned.source_variant=%r is not one of the configured Laya variants (%s)"
            % (cfg.laya_finetuned.source_variant, ", ".join(v.key for v in cfg.laya.variants))
        )
    checkpoint = Path(cfg.laya_finetuned.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not (checkpoint / "rl_agent_config.json").exists():
        raise RuntimeError(
            "fine-tuned checkpoint not found at %s. Run\n"
            "  python scripts/build_laya_finetune_data.py\n"
            "  python scripts/finetune_laya.py --data data/processed/laya_finetune_train.jsonl "
            "--output-dir %s\n"
            "or point models.laya_finetuned.checkpoint at an existing checkpoint."
            % (checkpoint, cfg.laya_finetuned.checkpoint)
        )
    variant = dataclasses.replace(
        source,
        key=FINETUNED_KEY,
        label=cfg.laya_finetuned.variant_label,
        note=(source.note + " " if source.note else "") +
             "Supervised fine-tuned on the Davidson train split (RLCD).",
        regime="fine-tuned",
    )
    laya_cfg = LayaConfig(
        repo=str(checkpoint),
        subfolder=None,
        revision=None,
        device=cfg.laya_finetuned.device,
        max_len=cfg.laya_finetuned.max_len,
        batch_size=cfg.laya_finetuned.batch_size,
        sort_by_length=cfg.laya_finetuned.sort_by_length,
        variants=[variant],
    )
    return LayaClassifier(laya_cfg, variant)


def build_models(cfg: Config, only: Optional[List[str]] = None) -> List[Tuple[str, Classifier]]:
    """Instantiate the enabled models and variants, skipping any name not in ``only``."""
    requested = set(only) if only else None
    models: List[Tuple[str, Classifier]] = []

    def add(key: str, classifier: Classifier) -> None:
        if requested is not None and key not in requested:
            return
        models.append((key, classifier))

    if cfg.majority.enabled:
        add("majority", MajorityClassifier())
    if cfg.tfidf.enabled:
        for variant in cfg.tfidf.variants:
            add(variant.key, TfidfLogisticClassifier(cfg.tfidf, variant))
    if cfg.laya.enabled:
        for variant in cfg.laya.variants:
            add(variant.key, LayaClassifier(cfg.laya, variant))
    if cfg.laya_finetuned.enabled:
        add(FINETUNED_KEY, build_finetuned_laya(cfg))
    if cfg.hatexplain.enabled:
        add("hatexplain", HateXplainClassifier(cfg.hatexplain))
    return models


def display_names(cfg: Config) -> Dict[str, str]:
    """Convenience map of key -> human name, without instantiating anything."""
    names: Dict[str, str] = {"majority": "Majority baseline", "hatexplain": "HateXplain BERT"}
    for variant in cfg.tfidf.variants:
        suffix = "" if not variant.class_weight else " (class_weight=%s)" % variant.class_weight
        names[variant.key] = "TF-IDF + Logistic Regression" + suffix
    for variant in cfg.laya.variants:
        names[variant.key] = "Laya (zero-shot, %s)" % variant.label
    if cfg.laya_finetuned.enabled:
        names[FINETUNED_KEY] = "Laya (fine-tuned, %s)" % cfg.laya_finetuned.variant_label
    return names
