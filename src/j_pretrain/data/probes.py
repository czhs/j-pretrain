"""Fixed evaluation probes.

A probe is a small, frozen set of packed sequences (exact token IDs) drawn
deterministically from a corpus's held-out validation split. The *same* probe
tokens are used for every checkpoint of every run, so cross-checkpoint loss
comparisons are apples-to-apples. Probes are stored as their own tiny packed
datasets plus a provenance manifest; they are never resampled per run.

This module only *builds and preserves* probes; running interpretability analysis
on them is explicitly out of scope for this reproduction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from j_pretrain.data.shards import PackedDataset, ShardWriter


def build_probe(
    source_val: PackedDataset,
    n_windows: int,
    out_dir: str | Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Copy the first ``n_windows`` windows of a val dataset into a frozen probe.

    Deterministic (prefix of the fixed val ordering). Fails loudly if the val
    split is too small for the requested probe size.
    """
    out_dir = Path(out_dir)
    if n_windows > len(source_val):
        raise ValueError(f"val split has {len(source_val)} windows < probe {n_windows}")
    w = ShardWriter(out_dir, seq_len=source_val.seq_len, shard_seqs=n_windows,
                    meta={**meta, "kind": "probe", "n_windows": n_windows})
    for i in range(n_windows):
        w.add(source_val[i])
    return w.finalize()


def write_probe_manifest(root: str | Path, entries: dict[str, dict[str, Any]]) -> None:
    """Write the top-level probe manifest listing every corpus probe + checksum."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "probe_manifest.json.tmp"
    tmp.write_text(json.dumps(entries, indent=2, sort_keys=True))
    os.replace(tmp, root / "probe_manifest.json")
