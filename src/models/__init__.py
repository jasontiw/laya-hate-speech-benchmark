"""Model registry for the benchmark."""
from __future__ import annotations

from typing import Dict, List, Tuple

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
    "build_models",
]

# Fixed order: the comparison table reads the same way on every run.
MODEL_ORDER: Tuple[str, ...] = ("majority", "tfidf", "laya", "hatexplain")


def build_models(cfg: Config, only: List[str] | None = None) -> List[Tuple[str, Classifier]]:
    """Instantiate the enabled models, skipping any name not in ``only``."""
    requested = set(only) if only else None
    builders = {
        "majority": lambda: MajorityClassifier(),
        "tfidf": lambda: TfidfLogisticClassifier(cfg.tfidf),
        "laya": lambda: LayaClassifier(cfg.laya),
        "hatexplain": lambda: HateXplainClassifier(cfg.hatexplain),
    }
    enabled: Dict[str, bool] = {
        "majority": cfg.majority.enabled,
        "tfidf": cfg.tfidf.enabled,
        "laya": cfg.laya.enabled,
        "hatexplain": cfg.hatexplain.enabled,
    }
    models: List[Tuple[str, Classifier]] = []
    for key in MODEL_ORDER:
        if not enabled.get(key, False):
            continue
        if requested is not None and key not in requested:
            continue
        models.append((key, builders[key]()))
    return models
