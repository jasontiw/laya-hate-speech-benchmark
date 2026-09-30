"""Model D — HateXplain BERT (PRD section 11).

``Hate-speech-CNERG/bert-base-uncased-hatexplain`` already classifies into three
classes. Its label order is read from the model config, never hardcoded, and then
mapped onto this project's canonical labels:

    hate speech -> hate speech
    offensive   -> offensive language
    normal      -> neither
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from ..config import HateXplainConfig
from ..environment import resolve_device
from .base import Classifier, Prediction


def map_hatexplain_label(raw: str) -> Optional[str]:
    """Map a HateXplain label string to a canonical label, by keyword."""
    value = str(raw).strip().lower()
    if "hate" in value:
        return "hate speech"
    if "offens" in value:
        return "offensive language"
    if "normal" in value or "neither" in value:
        return "neither"
    return None


class HateXplainClassifier(Classifier):
    key = "hatexplain"
    display_name = "HateXplain BERT"

    def __init__(self, cfg: HateXplainConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device: str | None = None
        self.index_to_label: Dict[int, str] = {}
        self.raw_labels: Dict[int, str] = {}

    def load(self, train_df: pd.DataFrame) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = resolve_device(self.cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, revision=self.cfg.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_id, revision=self.cfg.revision)
        self.model.to(self.device)
        self.model.eval()

        config_labels = {int(i): str(label) for i, label in self.model.config.id2label.items()}
        self.raw_labels = config_labels
        mapped: Dict[int, str] = {}
        for index, raw in config_labels.items():
            canonical = map_hatexplain_label(raw)
            if canonical:
                mapped[index] = canonical
        if len(mapped) != len(config_labels):
            raise RuntimeError(
                "could not map every HateXplain label to a canonical class: %r" % config_labels
            )
        self.index_to_label = mapped

    def predict_one(self, text: str) -> Prediction:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("hatexplain classifier used before load()")
        with self.torch.no_grad():
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.max_length,
            ).to(self.device)
            logits = self.model(**encoded).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1).cpu().tolist()
        by_label = {self.index_to_label[i]: float(p) for i, p in enumerate(probabilities)}
        best = max(by_label, key=by_label.get)
        return best, by_label

    def supports_probabilities(self) -> bool:
        return True

    def details(self) -> Dict[str, Any]:
        return {
            "model_id": self.cfg.model_id,
            "revision_requested": self.cfg.revision,
            "device": self.device,
            "max_length": self.cfg.max_length,
            "hatexplain_labels": {str(k): v for k, v in self.raw_labels.items()},
            "label_mapping": {str(k): v for k, v in self.index_to_label.items()},
        }
