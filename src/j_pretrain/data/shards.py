"""Memory-mapped packed-sequence shards with atomic writes and checksums.

A packed dataset is a directory of ``shard-NNNNN.npy`` files, each a 2-D
``uint16`` array of shape ``[n_seqs, seq_len]``, plus a ``manifest.json`` that
records every shard's checksum and the frozen provenance (dataset revision,
tokenizer hash, seq_len, split). Shards are memory-mapped at read time so the
full corpus is never resident in RAM.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from j_pretrain.data.packing import TOKEN_DTYPE

MANIFEST_NAME = "manifest.json"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class ShardWriter:
    """Accumulate fixed-length windows and flush them to atomic shard files."""

    out_dir: Path
    seq_len: int
    shard_seqs: int = 100_000  # windows per shard file
    meta: dict[str, Any] = field(default_factory=dict)
    _buf: list[np.ndarray] = field(default_factory=list)
    _shards: list[dict[str, Any]] = field(default_factory=list)
    _n_total: int = 0

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, window: np.ndarray) -> None:
        if window.shape != (self.seq_len,):
            raise ValueError(f"window shape {window.shape} != ({self.seq_len},)")
        self._buf.append(np.asarray(window, dtype=TOKEN_DTYPE))
        if len(self._buf) >= self.shard_seqs:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        idx = len(self._shards)
        arr = np.stack(self._buf, axis=0).astype(TOKEN_DTYPE)
        name = f"shard-{idx:05d}.npy"
        final = self.out_dir / name
        tmp = self.out_dir / (name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        self._shards.append({
            "name": name,
            "n_seqs": int(arr.shape[0]),
            "sha256": _sha256_file(final),
            "bytes": final.stat().st_size,
        })
        self._n_total += int(arr.shape[0])
        self._buf.clear()

    def finalize(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        self._flush()
        manifest = {
            "seq_len": self.seq_len,
            "dtype": np.dtype(TOKEN_DTYPE).name,
            "n_seqs": self._n_total,
            "n_tokens": self._n_total * self.seq_len,
            "shards": self._shards,
            **self.meta,
        }
        if extra:
            manifest.update(extra)
        tmp = self.out_dir / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(tmp, self.out_dir / MANIFEST_NAME)
        return manifest


class PackedDataset:
    """Read-only view over packed shards; shards are memory-mapped lazily."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / MANIFEST_NAME).read_text())
        self.seq_len = int(self.manifest["seq_len"])
        self._shard_meta = self.manifest["shards"]
        self._offsets = np.cumsum([0] + [s["n_seqs"] for s in self._shard_meta])
        self._n = int(self._offsets[-1])
        self._mmaps: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self._n

    def _mmap(self, shard_idx: int) -> np.ndarray:
        m = self._mmaps.get(shard_idx)
        if m is None:
            path = self.root / self._shard_meta[shard_idx]["name"]
            m = np.load(path, mmap_mode="r")
            self._mmaps[shard_idx] = m
        return m

    def __getitem__(self, i: int) -> np.ndarray:
        if i < 0:
            i += self._n
        if not (0 <= i < self._n):
            raise IndexError(i)
        shard_idx = int(np.searchsorted(self._offsets, i, side="right") - 1)
        local = i - int(self._offsets[shard_idx])
        return np.asarray(self._mmap(shard_idx)[local])

    def verify_checksums(self) -> bool:
        """Recompute every shard sha256 against the manifest; return True if all match."""
        for s in self._shard_meta:
            if _sha256_file(self.root / s["name"]) != s["sha256"]:
                return False
        return True
