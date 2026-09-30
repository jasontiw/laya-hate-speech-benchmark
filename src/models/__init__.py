"""Model registry for the benchmark.

The registry is built from the configuration rather than hardcoded, because a run
compares *variants* as well as models: TF-IDF with and without balanced class
weights, and several Laya prompt formulations. Each variant gets its own key, so
it becomes its own row in every artifact.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..config import Config
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


def available_model_keys(cfg: Config) -> List[str]:
    """The model keys this configuration would run, in reporting order."""
    keys: List[str] = []
    if cfg.majority.enabled:
        keys.append("majority")
    if cfg.tfidf.enabled:
        keys.extend(variant.key for variant in cfg.tfidf.variants)
    if cfg.laya.enabled:
        keys.extend(variant.key for variant in cfg.laya.variants)
    if cfg.hatexplain.enabled:
        keys.append("hatexplain")
    return keys


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
    return names
