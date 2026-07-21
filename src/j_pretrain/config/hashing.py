"""Deterministic configuration hashing and run-id derivation.

Config hashes make every run traceable to the exact frozen configuration. The
hash MUST be stable across processes/machines, so we serialize to canonical JSON
(sorted keys, fixed separators) before hashing. Run IDs are human-readable and
deterministic from the scientific identity of a run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _to_plain(obj: Any) -> Any:
    """Convert dataclasses / nested containers to plain JSON-able structures."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_plain(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, (str, bool, int, type(None))):
        return obj
    if isinstance(obj, float):
        # Normalize float repr so 1e-5 vs 0.00001 hash identically.
        return float(obj)
    # Fallback: stringify unknown types deterministically.
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Canonical, whitespace-stable JSON string for hashing."""
    plain = _to_plain(obj)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(obj: Any) -> str:
    """Full SHA-256 hex digest of an object's canonical JSON form."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short_hash(obj: Any, n: int = 8) -> str:
    """First ``n`` hex chars of :func:`config_hash`."""
    return config_hash(obj)[:n]


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (for checkpoint/dataset checksums)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fmt_lambda(lmbda: float) -> str:
    """Stable lambda token: 0.25 -> 'lambda-0.25', 1.0 -> 'lambda-1.0'."""
    s = ("%g" % float(lmbda))
    if "." not in s and "e" not in s:
        s = s + ".0"
    return f"lambda-{s}"


def derive_run_id(experiment: str, subset_tokens: int, lmbda: float) -> str:
    """Deterministic human-readable run id, e.g. 'music-300m_lambda-0.25'.

    The run id names the *pipeline* (Stage1->2->3 share one id); stage is a
    sub-path under the run's artifact tree. ``subset_tokens`` in tokens.
    """
    m = subset_tokens // 1_000_000
    return f"{experiment}-{m}m_{_fmt_lambda(lmbda)}"
