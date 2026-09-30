"""Model B — TF-IDF + Logistic Regression (PRD section 9).

Two variants are configured by default, because the dataset is heavily imbalanced
(offensive language is ~77% of the rows): the plain pipeline and the same pipeline
with ``class_weight="balanced"``. Comparing them separates "the model is
conservative about hate speech" from "the training prior is conservative".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import TASK_THREE_CLASS, TfidfConfig, TfidfVariant
from .base import Classifier, Prediction


class TfidfLogisticClassifier(Classifier):
    task = TASK_THREE_CLASS

    def __init__(self, cfg: TfidfConfig, variant: Optional[TfidfVariant] = None) -> None:
        self.cfg = cfg
        self.variant = variant or TfidfVariant()
        self.pipeline = None
        suffix = "" if not self.variant.class_weight else " (class_weight=%s)" % self.variant.class_weight
        self.key = self.variant.key
        self.display_name = "TF-IDF + Logistic Regression" + suffix

    def load(self, train_df: pd.DataFrame) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

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
                        class_weight=self.variant.class_weight or None,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.pipeline.fit(train_df["text"].tolist(), train_df["gold_label"].tolist())

    def _probabilities(self, text: str) -> Dict[str, float]:
        values = self.pipeline.predict_proba([text])[0]
        classes = [str(c) for c in self.pipeline.classes_]
        return {label: float(p) for label, p in zip(classes, values)}

    def predict_one(self, text: str) -> Prediction:
        if self.pipeline is None:
            raise RuntimeError("tfidf classifier used before load()")
        by_label = self._probabilities(text)
        return max(by_label, key=by_label.get), by_label

    def predict_batch(self, texts: List[str]) -> Optional[List[Prediction]]:
        # The vectoriser is already vectorised: one call over the whole list beats
        # a Python loop by a wide margin.
        if self.pipeline is None:
            raise RuntimeError("tfidf classifier used before load()")
        matrix = self.pipeline.predict_proba(texts)
        classes = [str(c) for c in self.pipeline.classes_]
        predictions: List[Prediction] = []
        for row in matrix:
            by_label = {label: float(p) for label, p in zip(classes, row)}
            predictions.append((max(by_label, key=by_label.get), by_label))
        return predictions

    def supports_probabilities(self) -> bool:
        return True

    def details(self) -> Dict[str, Any]:
        size = None
        if self.pipeline is not None:
            vectorizer = self.pipeline.named_steps["tfidf"]
            size = int(len(vectorizer.vocabulary_)) if hasattr(vectorizer, "vocabulary_") else None
        return {
            "variant": self.variant.key,
            "variant_note": self.variant.note,
            "task": self.task,
            "pipeline": "TfidfVectorizer -> LogisticRegression",
            "ngram_range": list(self.cfg.ngram_range),
            "min_df": self.cfg.min_df,
            "C": self.cfg.C,
            "class_weight": self.variant.class_weight,
            "vocabulary_size": size,
        }
