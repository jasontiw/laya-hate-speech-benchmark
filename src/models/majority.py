"""Model A — majority-class baseline (PRD section 8).

Reaches high accuracy on this imbalanced dataset while never once detecting hate
speech. It is the yardstick that makes every other number meaningful.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..config import TASK_THREE_CLASS
from .base import Classifier, Prediction


class MajorityClassifier(Classifier):
    key = "majority"
    display_name = "Majority baseline"
    task = TASK_THREE_CLASS

    def __init__(self) -> None:
        self.majority: str | None = None
        self.distribution: Dict[str, int] = {}

    def load(self, train_df: pd.DataFrame) -> None:
        counts = train_df["gold_label"].value_counts()
        self.distribution = {str(k): int(v) for k, v in counts.to_dict().items()}
        self.majority = str(counts.idxmax())

    def predict_one(self, text: str) -> Prediction:
        if self.majority is None:
            raise RuntimeError("majority classifier used before load()")
        return self.majority, None

    def details(self) -> Dict[str, Any]:
        return {"majority_class": self.majority, "train_class_distribution": self.distribution}
