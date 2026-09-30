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

# The canonical three classes, fixed forever in this order.
CANONICAL_LABELS: List[str] = ["hate speech", "offensive language", "neither"]
LABEL_INDEX: Dict[str, int] = {label: i for i, label in enumerate(CANONICAL_LABELS)}
HATE_LABEL = "hate speech"
NOT_HATE_LABELS: List[str] = ["offensive language", "neither"]

# Tasks a model can solve. A binary model answers hate-vs-rest only, so it has no
# three-class confusion matrix and is excluded from the three-class table.
TASK_THREE_CLASS = "three_class"
TASK_HATE_BINARY = "hate_binary"

# Results columns use a short slug per class (PRD section 13).
PROBABILITY_SLUG: Dict[str, str] = {
    "hate speech": "hate",
    "offensive language": "offensive",
    "neither": "neither",
}

# How each model relates to the Davidson training split. Reported so a reader never
# mistakes this benchmark for a controlled architecture comparison.
TRAINING_REGIME: Dict[str, Dict[str, Any]] = {
    "majority": {"regime": "Baseline", "trained_on_davidson": False,
                 "note": "Predicts the most frequent training label."},
    "tfidf": {"regime": "Supervised (fit here)", "trained_on_davidson": True,
              "note": "TF-IDF (1-2 grams) + Logistic Regression, fit on the train split."},
    "tfidf_balanced": {"regime": "Supervised (fit here)", "trained_on_davidson": True,
                       "note": "Same pipeline with class_weight='balanced'."},
    "laya": {"regime": "Zero-shot", "trained_on_davidson": False,
             "note": "Published checkpoint, hand-written question, no fine-tuning."},
    "laya_finetuned": {"regime": "Supervised fine-tuned (RLCD, fit here)", "trained_on_davidson": True,
                       "note": "Same L1 question, fine-tuned on the Davidson train split with RLCD "
                               "(scripts/finetune_laya.py); test rows never seen."},
    "hatexplain": {"regime": "Pretrained externally", "trained_on_davidson": False,
                   "note": "HateXplain BERT, trained on the HateXplain dataset, labels remapped."},
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
    # Carved out of the training side, so the test set is unchanged and every
    # number stays comparable with earlier runs.
    validation_size: float = 0.15
    seed: int = 42
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)


@dataclass
class MajorityConfig:
    enabled: bool = True


@dataclass
class TfidfVariant:
    key: str = "tfidf"
    class_weight: Optional[str] = None
    note: str = ""


@dataclass
class TfidfConfig:
    enabled: bool = True
    ngram_range: List[int] = field(default_factory=lambda: [1, 2])
    min_df: int = 2
    max_df: float = 1.0
    sublinear_tf: bool = True
    max_iter: int = 1000
    C: float = 1.0
    variants: List[TfidfVariant] = field(default_factory=lambda: [TfidfVariant()])


@dataclass
class LayaVariant:
    key: str = "laya"
    label: str = "current"
    task: str = TASK_THREE_CLASS
    question_type: str = "choice"
    instructions: str = "Classify the language of this message."
    criteria: Dict[str, str] = field(default_factory=dict)
    note: str = ""
    # Reported verbatim: "zero-shot" and "fine-tuned" are different experimental
    # conditions and the report must never let a reader confuse the two.
    regime: str = "zero-shot"


@dataclass
class LayaConfig:
    enabled: bool = True
    repo: str = "convaiinnovations/laya"
    subfolder: Optional[str] = None
    revision: Optional[str] = None
    device: Optional[str] = None
    max_len: int = 512
    # Batched inference (Laya's predict_batch): states per forward pass, and whether
    # to group similar lengths inside a pass.
    batch_size: int = 64
    sort_by_length: bool = True
    variants: List[LayaVariant] = field(default_factory=lambda: [LayaVariant()])


@dataclass
class LayaFinetunedConfig:
    """A Laya checkpoint produced by ``scripts/finetune_laya.py`` (phase 2A).

    Disabled by default: the checkpoint is a derived local artifact of ~0.8 GB, so a
    fresh clone must be able to run the zero-shot benchmark without it. Enable it with
    ``run_benchmark.py --include-finetuned`` once the fine-tune has been run.
    """

    enabled: bool = False
    # Directory holding model.safetensors, encoder/, tokenizer/, rl_agent_config.json.
    checkpoint: str = "artifacts/laya-finetuned-l1"
    # Zero-shot variant key whose question wording the fine-tune was trained on. The
    # fine-tuned row must answer the *same* question to be comparable.
    source_variant: str = "laya"
    variant_label: str = "L1 3-class choice (PRD wording)"
    device: Optional[str] = None
    max_len: int = 512
    batch_size: int = 64
    sort_by_length: bool = True


@dataclass
class HateXplainConfig:
    enabled: bool = True
    model_id: str = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    revision: Optional[str] = None
    device: Optional[str] = None
    max_length: int = 128
    batch_size: int = 64


@dataclass
class InferenceConfig:
    # "batch" feeds the whole test set through predict_batch (fast, order-preserving)
    # and measures single-item latency on a sample. "single" loops predict_one.
    predict_mode: str = "batch"
    latency_sample: int = 500
    warmup: int = 5


@dataclass
class StatsConfig:
    bootstrap_samples: int = 1000
    seed: int = 42
    confidence: float = 0.95


@dataclass
class CalibrationConfig:
    enabled: bool = True
    bins: int = 10
    # Fit Laya's own temperature map on the validation split. The shipped checkpoints are
    # over-confident, and temperature scaling cannot change a predicted label, only the
    # probabilities. See src/laya_calibration.py.
    fit_temperatures: bool = True
    seed: int = 42
    # Monotone maps for the hate score itself (Platt scaling, isotonic regression).
    # Fitted on validation, measured on test. They cannot change a ranking or a
    # prediction, only whether the probability means what it says.
    score_methods: List[str] = field(default_factory=lambda: ["platt", "isotonic"])


@dataclass
class OutputConfig:
    results_dir: str = "results"
    report_dir: str = "report"
    error_samples_per_category: int = 50
    error_examples_per_category: int = 3
    # "redacted" keeps verbatim tweet text out of the report, because those examples
    # are hate speech and the report may be published. The verbatim rows stay in the
    # git-ignored results/error_analysis.csv. Use "full" for a local-only report.
    error_examples_in_report: str = "redacted"


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    majority: MajorityConfig = field(default_factory=MajorityConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    laya: LayaConfig = field(default_factory=LayaConfig)
    laya_finetuned: LayaFinetunedConfig = field(default_factory=LayaFinetunedConfig)
    hatexplain: HateXplainConfig = field(default_factory=HateXplainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    raw: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None

    @property
    def results_dir(self) -> Path:
        return PROJECT_ROOT / self.output.results_dir

    @property
    def report_dir(self) -> Path:
        return PROJECT_ROOT / self.output.report_dir

    @property
    def inference_warmup(self) -> int:
        return self.inference.warmup


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
    tfidf_raw = dict(models.get("tfidf") or {})
    tfidf_declared = tfidf_raw.pop("variants", None)
    laya_raw = dict(models.get("laya") or {})
    laya_declared = laya_raw.pop("variants", None)

    cfg = Config(
        dataset=dataset,
        split=split,
        majority=_from(MajorityConfig, models.get("majority")),
        tfidf=_from(TfidfConfig, tfidf_raw),
        laya=_from(LayaConfig, laya_raw),
        laya_finetuned=_from(LayaFinetunedConfig, models.get("laya_finetuned")),
        hatexplain=_from(HateXplainConfig, models.get("hatexplain")),
        inference=_from(InferenceConfig, raw.get("inference")),
        stats=_from(StatsConfig, raw.get("stats")),
        calibration=_from(CalibrationConfig, raw.get("calibration")),
        output=_from(OutputConfig, raw.get("output")),
        raw=raw,
        path=str(config_path),
    )
    if tfidf_declared:
        cfg.tfidf.variants = [_from(TfidfVariant, item) for item in tfidf_declared]
    cfg.laya.variants = [_from(LayaVariant, item) for item in laya_declared] if laya_declared else [LayaVariant()]
    return cfg


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    """A JSON-serialisable view of the effective configuration."""
    data = dataclasses.asdict(cfg)
    data.pop("raw", None)
    data.pop("path", None)
    # asdict keeps the int keys of `labels`; JSON needs strings.
    data["dataset"]["labels"] = {str(k): v for k, v in cfg.dataset.labels.items()}
    return data
