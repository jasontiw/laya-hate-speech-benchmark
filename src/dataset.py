"""Dataset acquisition, verification and the deterministic grouped split.

Two rules drive this module:

* The raw file is never modified. It is downloaded once, hashed, and read-only.
* The split is deterministic and grouped: tweets that normalize to the same string
  always land on the same side, so a duplicated tweet cannot leak from train to test.
  Normalization is used *only* to find duplicates — the models still see the raw text.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CANONICAL_LABELS, Config, PROJECT_ROOT

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RT_RE = re.compile(r"^\s*rt\b[:\s]+", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

Logger = Callable[[str], None]


def _noop(_: str) -> None:  # pragma: no cover - default logger
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "laya-hate-speech-benchmark/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "wb") as out:
        out.write(response.read())
    return dest


def normalize(text: str, cfg) -> str:
    """A duplicate-detection key. Not shown to any model."""
    value = str(text)
    if cfg.lowercase:
        value = value.lower()
    if cfg.strip_urls:
        value = _URL_RE.sub(" ", value)
    if cfg.strip_mentions:
        value = _MENTION_RE.sub(" ", value)
    if cfg.strip_rt:
        value = _RT_RE.sub("", value)
    if cfg.strip_punctuation:
        value = _PUNCT_RE.sub(" ", value)
    if cfg.collapse_whitespace:
        value = _WS_RE.sub(" ", value).strip()
    return value


def load_raw(cfg: Config, log: Logger = _noop) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Download (if needed), verify and parse the Davidson dataset."""
    raw_path = PROJECT_ROOT / "data" / "raw" / "labeled_data.csv"
    if not raw_path.exists():
        if not cfg.dataset.url:
            raise RuntimeError("dataset.url is empty and no local copy exists at %s" % raw_path)
        log("dataset: downloading %s" % cfg.dataset.url)
        download(cfg.dataset.url, raw_path)

    digest = sha256_file(raw_path)
    verified: Optional[bool] = None
    if cfg.dataset.sha256:
        expected = cfg.dataset.sha256.strip().lower()
        verified = digest == expected
        if not verified:
            raise RuntimeError(
                "dataset SHA-256 mismatch: expected %s, got %s (%s). "
                "The upstream file changed; do not trust results until this is resolved." % (expected, digest, raw_path)
            )
    log("dataset: sha256=%s verified=%s" % (digest, verified))

    frame = pd.read_csv(raw_path)
    text_col, class_col = cfg.dataset.text_column, cfg.dataset.class_column
    if text_col not in frame.columns or class_col not in frame.columns:
        raise RuntimeError("expected columns %r and %r, found %r" % (text_col, class_col, list(frame.columns)))

    # The authors' CSV carries the original row index in an unnamed first column.
    id_col = frame.columns[0] if str(frame.columns[0]).startswith("Unnamed") else None
    out = pd.DataFrame(
        {
            "id": frame[id_col].astype(int) if id_col else np.arange(len(frame)),
            "text": frame[text_col].astype(str),
            "class_index": frame[class_col].astype(int),
        }
    )
    out = out.dropna(subset=["text", "class_index"])
    out["gold_label"] = out["class_index"].map(cfg.dataset.labels)
    missing = out["gold_label"].isna()
    if missing.any():
        raise RuntimeError("%d rows have a class index outside cfg.dataset.labels" % int(missing.sum()))
    out = out[["id", "text", "class_index", "gold_label"]].reset_index(drop=True)

    distribution = out["gold_label"].value_counts().to_dict()
    metadata: Dict[str, Any] = {
        "name": cfg.dataset.name,
        "description": cfg.dataset.description,
        "url": cfg.dataset.url,
        "file": str(raw_path),
        "sha256": digest,
        "sha256_expected": cfg.dataset.sha256,
        "sha256_verified": verified,
        "size_bytes": raw_path.stat().st_size,
        "rows": int(len(out)),
        "class_distribution": {label: int(distribution.get(label, 0)) for label in CANONICAL_LABELS},
        "file_modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(raw_path.stat().st_mtime)),
        "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log("dataset: %d rows, classes=%s" % (len(out), metadata["class_distribution"]))
    return out, metadata


def build_split(df: pd.DataFrame, cfg: Config, log: Logger = _noop) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Stratified, duplicate-grouped train/test split (PRD section 7)."""
    frame = df.copy()
    frame["norm_text"] = [normalize(t, cfg.split.normalize) for t in frame["text"]]
    # Factorize gives every distinct normalized tweet its own group id.
    frame["group_id"] = pd.factorize(frame["norm_text"], sort=False)[0]

    # A group's class is the majority class of its members (duplicates rarely disagree).
    group_label = frame.groupby("group_id")["gold_label"].agg(lambda s: s.value_counts().idxmax())

    rng = np.random.default_rng(cfg.split.seed)
    test_groups: set[int] = set()
    per_class_groups: Dict[str, int] = {}
    for label in CANONICAL_LABELS:
        # np.array(...) copies: Index.to_numpy() can return a read-only view (pandas CoW),
        # and rng.shuffle writes in place.
        groups = np.array(group_label[group_label == label].index, dtype=np.int64)
        rng.shuffle(groups)
        per_class_groups[label] = int(groups.size)
        n_test = int(round(groups.size * cfg.split.test_size))
        test_groups.update(int(g) for g in groups[:n_test])

    frame["split"] = np.where(frame["group_id"].isin(test_groups), "test", "train")

    train_ids = sorted(int(i) for i in frame.loc[frame["split"] == "train", "id"])
    test_ids = sorted(int(i) for i in frame.loc[frame["split"] == "test", "id"])
    signature = hashlib.sha256((",".join(str(i) for i in test_ids)).encode("utf-8")).hexdigest()

    def _class_counts(part: pd.DataFrame) -> Dict[str, int]:
        counts = part["gold_label"].value_counts().to_dict()
        return {label: int(counts.get(label, 0)) for label in CANONICAL_LABELS}

    train_part = frame[frame["split"] == "train"]
    test_part = frame[frame["split"] == "test"]
    info: Dict[str, Any] = {
        "seed": cfg.split.seed,
        "test_size": cfg.split.test_size,
        "grouping": "normalized text (duplicate tweets kept in the same split)",
        "normalize": {
            "lowercase": cfg.split.normalize.lowercase,
            "strip_urls": cfg.split.normalize.strip_urls,
            "strip_mentions": cfg.split.normalize.strip_mentions,
            "strip_rt": cfg.split.normalize.strip_rt,
            "strip_punctuation": cfg.split.normalize.strip_punctuation,
            "collapse_whitespace": cfg.split.normalize.collapse_whitespace,
        },
        "total_rows": int(len(frame)),
        "total_groups": int(frame["group_id"].nunique()),
        "groups_per_class": per_class_groups,
        "train_rows": int(len(train_part)),
        "test_rows": int(len(test_part)),
        "train_class_counts": _class_counts(train_part),
        "test_class_counts": _class_counts(test_part),
        "test_id_sha256": signature,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }
    log(
        "split: train=%d test=%d groups=%d signature=%s"
        % (len(train_part), len(test_part), info["total_groups"], signature[:12])
    )
    return frame, info


def save_processed(frame: pd.DataFrame, log: Logger = _noop) -> Path:
    path = PROJECT_ROOT / "data" / "processed" / "dataset_split.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[["id", "text", "class_index", "gold_label", "group_id", "split"]].to_csv(path, index=False)
    log("dataset: processed split -> %s" % path)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
