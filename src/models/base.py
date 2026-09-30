"""Common model interface.

Two things every model declares:

* its ``task`` - ``three_class`` (hate / offensive / neither) or ``hate_binary``
  (hate vs rest). A binary model has no three-class confusion matrix, so it is
  excluded from the three-class tables instead of being scored on labels it never
  emits.
* whether it can batch (:meth:`predict_batch`). Predictions always come from one
  authoritative path; batching is used for throughput and cross-checked against the
  single-item path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import TASK_THREE_CLASS

# (predicted label, optional {label -> probability})
Prediction = Tuple[str, Optional[Dict[str, float]]]


@dataclass
class ModelResult:
    """Everything a model produced, plus how long each part took."""

    key: str
    name: str
    labels: List[str]
    task: str = TASK_THREE_CLASS
    probabilities: Optional[List[Optional[Dict[str, float]]]] = None
    latencies_ms: List[float] = field(default_factory=list)
    load_time_s: float = 0.0
    device: Optional[str] = None
    revision: Optional[str] = None
    batch: Optional[Dict[str, Any]] = None
    token_stats: Optional[Dict[str, Any]] = None
    calibrated_labels: Optional[List[str]] = None
    calibrated_probabilities: Optional[List[Optional[Dict[str, float]]]] = None
    calibration_fit: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)


class Classifier(ABC):
    """A benchmark model. Subclasses implement :meth:`load` and :meth:`predict_one`."""

    key: str = "classifier"
    display_name: str = "classifier"
    task: str = TASK_THREE_CLASS
    # True only for models that can fit a post-hoc calibration map.
    supports_calibration: bool = False

    def load(self, train_df: pd.DataFrame) -> None:
        """Fit on the training split or load pretrained weights. Called once."""
        return None

    @abstractmethod
    def predict_one(self, text: str) -> Prediction:
        """Classify a single tweet."""

    def predict_batch(self, texts: List[str]) -> Optional[List[Prediction]]:
        """Classify many tweets at once, in input order.

        Returns None when the model has no batched path, so the caller falls back to
        looping :meth:`predict_one` without special-casing the model.
        """
        return None

    def fit_calibration(self, states: List[Any], gold_labels: List[str], seed: int = 42) -> Optional[Dict[str, Any]]:
        """Fit a post-hoc calibration map from labeled data. None when unsupported."""
        return None

    def save_calibration(self, path) -> None:
        """Persist a fitted calibration map, when the model has one."""
        return None

    def supports_probabilities(self) -> bool:
        return False

    def details(self) -> Dict[str, Any]:
        return {}
