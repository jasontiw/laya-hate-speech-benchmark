"""Model B — TF-IDF + Logistic Regression (PRD section 9).

A classical, cheap, well-understood baseline: how much does a modern decision
model add over a bag-of-n-grams linear classifier?
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..config import TfidfConfig
from .base import Classifier, Prediction


class TfidfLogisticClassifier(Classifier):
    key = "tfidf"
    display_name = "TF-IDF + Logistic Regression"

    def __init__(self, cfg: TfidfConfig) -> None:
        self.cfg = cfg
        self.pipeline = None

    def load(self, train_df: pd.DataFrame) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        class_weight = self.cfg.class_weight or None
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=tuple(self.cfg.ngram_range),
                        min_df=self.cfg.min_df,
                        max_df=self.cfg.max_df,
                        sublinear_tf=self.cfg.sublinear_tf,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=self.cfg.max_iter,
                        C=self.cfg.C,
                        class_weight=class_weight,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.pipeline.fit(train_df["text"].tolist(), train_df["gold_label"].tolist())

    def predict_one(self, text: str) -> Prediction:
        if self.pipeline is None:
            raise RuntimeError("tfidf classifier used before load()")
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = [str(c) for c in self.pipeline.classes_]
        by_label = {label: float(p) for label, p in zip(classes, probabilities)}
        best = max(by_label, key=by_label.get)
        return best, by_label

    def supports_probabilities(self) -> bool:
        return True

    def details(self) -> Dict[str, Any]:
        size = None
        if self.pipeline is not None:
            vectorizer = self.pipeline.named_steps["tfidf"]
            size = int(len(vectorizer.vocabulary_)) if hasattr(vectorizer, "vocabulary_") else None
        return {
            "pipeline": "TfidfVectorizer -> LogisticRegression",
            "ngram_range": list(self.cfg.ngram_range),
            "min_df": self.cfg.min_df,
            "C": self.cfg.C,
            "class_weight": self.cfg.class_weight,
            "vocabulary_size": size,
        }
