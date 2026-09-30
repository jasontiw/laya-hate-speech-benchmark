"""Model D — HateXplain BERT (PRD section 11).

``Hate-speech-CNERG/bert-base-uncased-hatexplain`` already classifies into three
classes. Its label order is read from the model config, never hardcoded, and then
mapped onto this project's canonical labels:

    hate speech -> hate speech
    offensive   -> offensive language
    normal      -> neither
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import TASK_THREE_CLASS, HateXplainConfig
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
    task = TASK_THREE_CLASS

    def __init__(self, cfg: HateXplainConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device: str | None = None
        self.index_to_label: Dict[int, str] = {}
        self.raw_labels: Dict[int, str] = {}
        self._truncated = 0
        self._rows = 0
        self._max_tokens = 0

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
            raise RuntimeError("could not map every HateXplain label to a canonical class: %r" % config_labels)
        self.index_to_label = mapped

    def _encode(self, texts: List[str]):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_length,
            padding=True,
        )
        lengths = [int(v) for v in encoded["attention_mask"].sum(dim=1).tolist()]
        self._rows += len(texts)
        self._truncated += sum(1 for length in lengths if length >= self.cfg.max_length)
        if lengths:
            self._max_tokens = max(self._max_tokens, max(lengths))
        encoded.pop("token_type_ids", None)
        return encoded.to(self.device)

    def _decode(self, logits) -> List[Prediction]:
        probabilities = self.torch.softmax(logits, dim=-1).cpu().tolist()
        predictions: List[Prediction] = []
        for row in probabilities:
            by_label = {self.index_to_label[i]: float(p) for i, p in enumerate(row)}
            predictions.append((max(by_label, key=by_label.get), by_label))
        return predictions

    def predict_one(self, text: str) -> Prediction:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("hatexplain classifier used before load()")
        with self.torch.no_grad():
            logits = self.model(**self._encode([text])).logits
        return self._decode(logits)[0]

    def predict_batch(self, texts: List[str]) -> Optional[List[Prediction]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("hatexplain classifier used before load()")
        predictions: List[Prediction] = []
        size = max(1, int(self.cfg.batch_size or 1))
        with self.torch.no_grad():
            for start in range(0, len(texts), size):
                chunk = texts[start:start + size]
                logits = self.model(**self._encode(list(chunk))).logits
                predictions.extend(self._decode(logits))
        return predictions

    def supports_probabilities(self) -> bool:
        return True

    def token_stats(self) -> Dict[str, Any]:
        return {
            "rows": self._rows,
            "truncated_rows": self._truncated,
            "max_state_tokens": self._max_tokens or None,
            "max_length": self.cfg.max_length,
        }

    def reset_stats(self) -> None:
        """Clear token/truncation counters, so the reported stats cover one pass only."""
        self._rows = 0
        self._truncated = 0
        self._max_tokens = 0

    def details(self) -> Dict[str, Any]:
        return {
            "model_id": self.cfg.model_id,
            "revision_requested": self.cfg.revision,
            "device": self.device,
            "task": self.task,
            "max_length": self.cfg.max_length,
            "batch_size": self.cfg.batch_size,
            "hatexplain_labels": {str(k): v for k, v in self.raw_labels.items()},
            "label_mapping": {str(k): v for k, v in self.index_to_label.items()},
        }
