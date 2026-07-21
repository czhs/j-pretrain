"""Deterministic, document-level train / tune / val split assignment.

Each document is assigned to exactly one split by hashing a stable document key
(salted per corpus). This guarantees:

* **Determinism** — the same key always maps to the same split, on any machine.
* **Disjointness** — a document is in exactly one split; no document overlap.
* **No packing leakage** — because splitting is at the document level *before*
  packing, no seq_len window can straddle two splits.

Ratios are expressed in per-mille (out of 1000) buckets so counts are exact.
For C4 we additionally have a native ``validation`` split available; the split
policy is recorded in ``docs/DATA_AUDIT.md``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

SPLIT_TRAIN = "train"
SPLIT_TUNE = "tune"
SPLIT_VAL = "val"


@dataclass(frozen=True)
class SplitPolicy:
    """Per-mille bucket bands. val = [0, val_pm); tune = [val_pm, val_pm+tune_pm); rest train."""

    val_pm: int = 5   # 0.5%
    tune_pm: int = 5  # 0.5%
    salt: str = ""

    def __post_init__(self) -> None:
        if self.val_pm < 0 or self.tune_pm < 0 or self.val_pm + self.tune_pm >= 1000:
            raise ValueError("invalid split per-mille bands")


def _bucket(key: str, salt: str) -> int:
    h = hashlib.sha256(f"{salt}\x00{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % 1000


def split_of(key: str, policy: SplitPolicy) -> str:
    """Return the split name for a stable document key under ``policy``."""
    b = _bucket(key, policy.salt)
    if b < policy.val_pm:
        return SPLIT_VAL
    if b < policy.val_pm + policy.tune_pm:
        return SPLIT_TUNE
    return SPLIT_TRAIN


def doc_key(source: str, revision: str, doc_id: str) -> str:
    """Stable, corpus-scoped document key used for splitting and provenance."""
    return f"{source}@{revision}:{doc_id}"
