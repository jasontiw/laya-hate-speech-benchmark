"""Model C — Laya, zero-shot (PRD section 10).

Hate-speech detection is posed as a Laya ``choice`` question with three options.
No training happens here: the model sees the question wording and the candidate
labels, exactly as an application would supply them.

The module is named ``laya_model`` rather than ``laya`` so it can never shadow the
installed ``laya`` package on import (see the Laya repository's own AGENTS.md note
about a local ``laya/`` directory shadowing modules).
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ..config import LayaConfig
from .base import Classifier, Prediction


class LayaClassifier(Classifier):
    key = "laya"
    display_name = "Laya (zero-shot)"

    def __init__(self, cfg: LayaConfig) -> None:
        self.cfg = cfg
        self.agent = None
        self.question: Dict[str, Any] | None = None
        self.laya_version: str | None = None
        self.device: str | None = None
        self.revision: str | None = None

    def load(self, train_df: pd.DataFrame) -> None:
        import laya

        self.laya_version = getattr(laya, "__version__", None)
        self.agent = laya.load(
            self.cfg.repo,
            subfolder=self.cfg.subfolder or None,
            device=self.cfg.device or None,
            revision=self.cfg.revision or None,
        )
        self.device = getattr(self.agent, "device", None)
        self.revision = getattr(self.agent, "revision", None)
        # Laya reports a torch.device object here; keep metadata JSON-serialisable.
        self.device = str(self.device) if self.device is not None else None
        self.revision = str(self.revision) if self.revision is not None else None
        self.question = {
            "label": {
                "type": "choice",
                "instructions": self.cfg.instructions,
                "criteria": dict(self.cfg.criteria),
            }
        }

    def predict_one(self, text: str) -> Prediction:
        if self.agent is None or self.question is None:
            raise RuntimeError("laya classifier used before load()")
        try:
            result = self.agent.predict(text, self.question, max_len=self.cfg.max_len)
        except TypeError:
            # Older/newer signature without max_len: fall back to the default budget.
            result = self.agent.predict(text, self.question)
        answer = result["answers"]["label"]
        probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
        return str(answer["choice"]), probabilities

    def supports_probabilities(self) -> bool:
        return True

    def details(self) -> Dict[str, Any]:
        return {
            "repo": self.cfg.repo,
            "subfolder": self.cfg.subfolder,
            "revision_requested": self.cfg.revision,
            "revision_resolved": self.revision,
            "laya_version": self.laya_version,
            "device": self.device,
            "max_len": self.cfg.max_len,
            "questions": self.question,
        }
