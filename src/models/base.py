"""Common model interface.

Every model is loaded once, warmed up, then asked to classify one tweet at a time.
Keeping prediction per-item is deliberate: it is what makes the reported p50/p95
latency an honest per-example latency rather than a per-batch one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# (predicted canonical label, optional {canonical label -> probability})
Prediction = Tuple[str, Optional[Dict[str, float]]]


@dataclass
class ModelResult:
    """Everything a model produced, plus how long each part took."""

    key: str
    name: str
    labels: List[str]
    probabilities: Optional[List[Optional[Dict[str, float]]]] = None
    latencies_ms: List[float] = field(default_factory=list)
    load_time_s: float = 0.0
    device: Optional[str] = None
    revision: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def supports_probabilities(self) -> bool:
        return self.probabilities is not None


class Classifier(ABC):
    """A benchmark model. Subclasses implement :meth:`load` and :meth:`predict_one`."""

    key: str = "classifier"
    display_name: str = "classifier"

    def load(self, train_df: pd.DataFrame) -> None:
        """Fit on the training split or load pretrained weights. Called once."""
        return None

    @abstractmethod
    def predict_one(self, text: str) -> Prediction:
        """Classify a single tweet, returning a canonical label."""

    def supports_probabilities(self) -> bool:
        return False

    def details(self) -> Dict[str, Any]:
        return {}
