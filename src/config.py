"""Typed configuration for the hate-speech benchmark.

``config.yaml`` at the project root is the source of truth. This module parses it
into dataclasses, applies defaults, and can serialise it back to a plain dict for
the run artifacts.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

# The canonical three classes, fixed forever in this order. Every model's output
# is mapped onto them (see src/models/hatexplain.py for the HateXplain mapping).
CANONICAL_LABELS: List[str] = ["hate speech", "offensive language", "neither"]
LABEL_INDEX: Dict[str, int] = {label: i for i, label in enumerate(CANONICAL_LABELS)}
HATE_LABEL = "hate speech"

# Results columns use a short slug per class (PRD section 13).
PROBABILITY_SLUG: Dict[str, str] = {
    "hate speech": "hate",
    "offensive language": "offensive",
    "neither": "neither",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

T = TypeVar("T")


@dataclass
class DatasetConfig:
    name: str = "tdavidson/hate_speech_offensive"
    description: str = ""
    url: str = ""
    sha256: Optional[str] = None
    text_column: str = "tweet"
    class_column: str = "class"
    labels: Dict[int, str] = field(default_factory=lambda: {i: l for i, l in enumerate(CANONICAL_LABELS)})


@dataclass
class NormalizeConfig:
    lowercase: bool = True
    strip_urls: bool = True
    strip_mentions: bool = True
    strip_rt: bool = True
    strip_punctuation: bool = False
    collapse_whitespace: bool = True


@dataclass
class SplitConfig:
    test_size: float = 0.2
    seed: int = 42
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)


@dataclass
class MajorityConfig:
    enabled: bool = True


@dataclass
class TfidfConfig:
    enabled: bool = True
    ngram_range: List[int] = field(default_factory=lambda: [1, 2])
    min_df: int = 2
    max_df: float = 1.0
    sublinear_tf: bool = True
    max_iter: int = 1000
    C: float = 1.0
    class_weight: Optional[str] = None


@dataclass
class LayaConfig:
    enabled: bool = True
    repo: str = "convaiinnovations/laya"
    subfolder: Optional[str] = None
    revision: Optional[str] = None
    device: Optional[str] = None
    max_len: int = 512
    instructions: str = "Classify the language of this message."
    criteria: Dict[str, str] = field(default_factory=dict)


@dataclass
class HateXplainConfig:
    enabled: bool = True
    model_id: str = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    revision: Optional[str] = None
    device: Optional[str] = None
    max_length: int = 128


@dataclass
class OutputConfig:
    results_dir: str = "results"
    report_dir: str = "report"
    error_samples_per_category: int = 50


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    majority: MajorityConfig = field(default_factory=MajorityConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    laya: LayaConfig = field(default_factory=LayaConfig)
    hatexplain: HateXplainConfig = field(default_factory=HateXplainConfig)
    inference_warmup: int = 5
    output: OutputConfig = field(default_factory=OutputConfig)
    raw: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None

    @property
    def results_dir(self) -> Path:
        return PROJECT_ROOT / self.output.results_dir

    @property
    def report_dir(self) -> Path:
        return PROJECT_ROOT / self.output.report_dir


def _from(cls: Type[T], data: Any) -> T:
    """Build a dataclass from a mapping, ignoring unknown keys."""
    data = data or {}
    allowed = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def _labels(raw: Any) -> Dict[int, str]:
    if not raw:
        return {i: label for i, label in enumerate(CANONICAL_LABELS)}
    return {int(k): str(v) for k, v in dict(raw).items()}


def load_config(path: Optional[str] = None) -> Config:
    """Load ``config.yaml`` (or ``path``) into a :class:`Config`."""
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    ds_raw = raw.get("dataset") or {}
    dataset = _from(DatasetConfig, ds_raw)
    dataset.labels = _labels(ds_raw.get("labels"))

    split_raw = raw.get("split") or {}
    split = _from(SplitConfig, split_raw)
    split.normalize = _from(NormalizeConfig, split_raw.get("normalize"))

    models = raw.get("models") or {}
    cfg = Config(
        dataset=dataset,
        split=split,
        majority=_from(MajorityConfig, models.get("majority")),
        tfidf=_from(TfidfConfig, models.get("tfidf")),
        laya=_from(LayaConfig, models.get("laya")),
        hatexplain=_from(HateXplainConfig, models.get("hatexplain")),
        inference_warmup=int((raw.get("inference") or {}).get("warmup", 5)),
        output=_from(OutputConfig, raw.get("output")),
        raw=raw,
        path=str(config_path),
    )
    return cfg


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    """A JSON-serialisable view of the effective configuration."""
    data = dataclasses.asdict(cfg)
    data.pop("raw", None)
    data.pop("path", None)
    # asdict keeps the int keys of `labels`; JSON needs strings.
    data["dataset"]["labels"] = {str(k): v for k, v in cfg.dataset.labels.items()}
    return data
