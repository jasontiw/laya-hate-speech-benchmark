"""Fit Laya's own temperature map on the validation split.

Why this exists: the benchmark measured Laya's hate-score calibration as bad
(ECE 0.21-0.42, and `noul` with a Brier score worse than a constant predictor at the
base rate). Laya ships the machinery to fix it — `Agent.fit_temperatures` /
`save_calibration` / `laya.load(..., calibration=...)` — and its own documentation says
the published checkpoints are over-confident as shipped. This module wires that
machinery to the benchmark's validation split.

Two properties worth stating because they shape what the report can claim:

* the fit happens on **validation**; the effect is measured on **test**,
* temperature scaling is monotone, so it **cannot change a predicted label** — only the
  probabilities move. The report checks that invariant instead of assuming it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .config import HATE_LABEL

NOT_HATE = "not hate speech"


def option_index(question_type: str, criteria: Optional[Dict[str, str]], gold_label: str) -> int:
    """Where the gold label sits in the option list, in the order Laya renders options.

    ``render_options`` documents the contract this mirrors: ``choice`` options follow the
    ``criteria`` dict insertion order, and ``noul`` is always ``[false, true]``.
    """
    if question_type == "noul":
        return 1 if str(gold_label) == HATE_LABEL else 0
    keys = list((criteria or {}).keys())
    if not keys:
        raise ValueError("choice questions need criteria to locate the gold option")
    if gold_label in keys:
        return keys.index(gold_label)
    if NOT_HATE in keys:
        return keys.index(NOT_HATE)
    raise ValueError("gold label %r is not among the options %r" % (gold_label, keys))


def target_vector(index: int, size: int) -> List[float]:
    return [1.0 if position == index else 0.0 for position in range(size)]


def build_pairs(
    states: Sequence[Any],
    gold_labels: Sequence[str],
    question: Dict[str, Any],
    question_type: str,
    criteria: Optional[Dict[str, str]],
) -> List[Any]:
    """``(state, questions, targets)`` triples for ``laya.calibrate.records_from_labeled``."""
    question_id = next(iter(question))
    size = 2 if question_type == "noul" else len(criteria or {})
    pairs = []
    for state, gold in zip(states, gold_labels):
        index = option_index(question_type, criteria, gold)
        pairs.append((state, question, {question_id: target_vector(index, size)}))
    return pairs


def fit(agent, pairs: Sequence[Any], seed: int = 42) -> Dict[str, Any]:
    """Fit and install a temperature map on ``agent`` from labeled validation forwards.

    The ``torch.no_grad()`` wrapper works around a bug in laya 0.3.22:
    ``laya.calibrate.records_from_labeled`` calls ``agent._forward()`` directly, but only
    ``predict``/``predict_batch`` carry ``@torch.no_grad()``. On a model whose parameters
    require grad, ``_forward`` then raises
    ``RuntimeError: Can't call numpy() on Tensor that requires grad``. The fix belongs
    upstream (decorate ``records_from_labeled``, or ``_forward``); the wrapper is
    behaviour-preserving and safe to drop once that lands.
    """
    import torch
    from laya.calibrate import records_from_labeled

    with torch.no_grad():
        records = records_from_labeled(agent, pairs)
    return agent.fit_temperatures(records, compute_ece=True, seed=seed)
