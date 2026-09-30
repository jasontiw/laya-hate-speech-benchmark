"""Model C — Laya, zero-shot (PRD section 10).

Hate-speech detection is posed as a Laya question. No training happens here: the
model sees the question wording and the candidate labels, exactly as an application
would supply them.

Variants let the experiment answer *which formulation* works, instead of assuming:

* ``question_type: choice`` + task ``three_class`` - hate / offensive / neither,
* ``question_type: choice`` + task ``hate_binary`` - hate / not hate (2 options),
* ``question_type: noul``   + task ``hate_binary`` - a calibrated P(hate) you threshold.

Predictions use Laya's batched path when available (``predict_batch``); the
single-item path is measured separately for latency, so throughput and per-item
latency are never mixed.

The module is named ``laya_model`` rather than ``laya`` so it can never shadow the
installed ``laya`` package on import.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import HATE_LABEL, TASK_HATE_BINARY, TASK_THREE_CLASS, LayaConfig, LayaVariant
from .base import Classifier, Prediction

NOT_HATE = "not hate speech"


class LayaClassifier(Classifier):
    supports_calibration = True

    def __init__(self, cfg: LayaConfig, variant: Optional[LayaVariant] = None) -> None:
        self.cfg = cfg
        self.variant = variant or LayaVariant()
        self.task = self.variant.task
        self.key = self.variant.key
        self.display_name = "Laya (zero-shot, %s)" % self.variant.label
        self.agent = None
        self.question: Dict[str, Any] | None = None
        self.laya_version: str | None = None
        self.device: str | None = None
        self.revision: str | None = None
        self._usage = {"rows": 0, "truncated": 0, "sum_state_tokens": 0, "max_state_tokens": 0}

    # ------------------------------------------------------------------ setup
    def load(self, train_df: pd.DataFrame) -> None:
        import laya

        self.laya_version = getattr(laya, "__version__", None)
        self.agent = laya.load(
            self.cfg.repo,
            subfolder=self.cfg.subfolder or None,
            device=self.cfg.device or None,
            revision=self.cfg.revision or None,
        )
        self.device = str(self.agent.device) if getattr(self.agent, "device", None) is not None else None
        self.revision = str(self.agent.revision) if getattr(self.agent, "revision", None) is not None else None
        self.question = self._build_question()

    def _build_question(self) -> Dict[str, Any]:
        if self.variant.question_type == "noul":
            question: Dict[str, Any] = {
                "is_hate": {"type": "noul", "instructions": self.variant.instructions}
            }
            if self.variant.criteria:
                question["is_hate"]["criteria"] = dict(self.variant.criteria)
            return question
        return {
            "label": {
                "type": "choice",
                "instructions": self.variant.instructions,
                "criteria": dict(self.variant.criteria),
            }
        }

    # -------------------------------------------------------------- inference
    def _record_usage(self, result: Dict[str, Any]) -> None:
        usage = (result or {}).get("usage") or {}
        self._usage["rows"] += 1
        if usage.get("truncated"):
            self._usage["truncated"] += 1
        tokens = usage.get("state_tokens")
        if isinstance(tokens, int):
            self._usage["sum_state_tokens"] += tokens
            self._usage["max_state_tokens"] = max(self._usage["max_state_tokens"], tokens)

    def _parse(self, result: Dict[str, Any]) -> Prediction:
        answers = result["answers"]
        answer = answers.get("is_hate") or next(iter(answers.values()))
        if str(answer.get("type")) == "noul" or "noul" in answer:
            value = float(answer["noul"])
            label = HATE_LABEL if value >= 0.5 else NOT_HATE
            # A noul answer is a probability; expose it as a two-label distribution so
            # every downstream metric (PR-AUC, thresholding, calibration) sees the same shape.
            return label, {HATE_LABEL: value, NOT_HATE: 1.0 - value}
        probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
        return str(answer["choice"]), probabilities

    def predict_one(self, text: str) -> Prediction:
        if self.agent is None or self.question is None:
            raise RuntimeError("laya classifier used before load()")
        try:
            result = self.agent.predict(text, self.question, max_len=self.cfg.max_len)
        except TypeError:
            result = self.agent.predict(text, self.question)
        self._record_usage(result)
        return self._parse(result)

    def predict_batch(self, texts: List[str]) -> Optional[List[Prediction]]:
        if self.agent is None or self.question is None:
            raise RuntimeError("laya classifier used before load()")
        try:
            results = self.agent.predict_batch(
                list(texts),
                self.question,
                batch_size=self.cfg.batch_size,
                sort_by_length=self.cfg.sort_by_length,
                max_len=self.cfg.max_len,
            )
        except TypeError:
            results = self.agent.predict_batch(list(texts), self.question, batch_size=self.cfg.batch_size)
        for result in results:
            self._record_usage(result)
        return [self._parse(result) for result in results]

    def supports_probabilities(self) -> bool:
        return True

    def fit_calibration(self, states: List[Any], gold_labels: List[str], seed: int = 42) -> Optional[Dict[str, Any]]:
        """Fit Laya's temperature map from labeled validation forwards (installs it too)."""
        if self.agent is None or self.question is None:
            raise RuntimeError("laya classifier used before load()")
        from ..laya_calibration import build_pairs, fit as fit_temperatures

        pairs = build_pairs(states, gold_labels, self.question, self.variant.question_type, self.variant.criteria)
        return fit_temperatures(self.agent, pairs, seed=seed)

    def save_calibration(self, path) -> None:
        if self.agent is not None:
            self.agent.save_calibration(str(path))

    def token_stats(self) -> Dict[str, Any]:
        rows = max(1, self._usage["rows"])
        return {
            "rows": self._usage["rows"],
            "truncated_rows": self._usage["truncated"],
            "mean_state_tokens": self._usage["sum_state_tokens"] / rows if self._usage["rows"] else None,
            "max_state_tokens": self._usage["max_state_tokens"] or None,
            "max_len": self.cfg.max_len,
        }

    def reset_stats(self) -> None:
        """Clear token/truncation counters, so the reported stats cover one pass only."""
        self._usage = {"rows": 0, "truncated": 0, "sum_state_tokens": 0, "max_state_tokens": 0}

    def details(self) -> Dict[str, Any]:
        return {
            "variant": self.variant.key,
            "variant_label": self.variant.label,
            "variant_note": self.variant.note,
            "task": self.task,
            "question_type": self.variant.question_type,
            "repo": self.cfg.repo,
            "subfolder": self.cfg.subfolder,
            "revision_requested": self.cfg.revision,
            "revision_resolved": self.revision,
            "laya_version": self.laya_version,
            "device": self.device,
            "max_len": self.cfg.max_len,
            "batch_size": self.cfg.batch_size,
            "sort_by_length": self.cfg.sort_by_length,
            "questions": self.question,
        }

    def unload(self) -> None:
        """Drop the checkpoint so N variants do not all stay resident."""
        self.agent = None
