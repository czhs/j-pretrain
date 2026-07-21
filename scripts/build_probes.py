#!/usr/bin/env python
"""Build the fixed evaluation probes (one per corpus) from the packed val splits.

A probe is the first ``n_windows`` windows of a corpus's frozen validation
:class:`PackedDataset` (deterministic prefix of the fixed val ordering), copied
into its own tiny packed dataset plus a provenance manifest. The SAME probe tokens
are used for every checkpoint of every run, so cross-checkpoint loss comparisons are
apples-to-apples. Probes are built ONCE and never resampled per run.

Run after ``scripts/build_datasets.py`` finishes (the six val manifests must exist):

    J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts \
    /home/hshi-j-4090/miniconda3/envs/jpre/bin/python scripts/build_probes.py

Idempotent: an existing complete probe manifest is left untouched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from j_pretrain.data.probes import build_probe, write_probe_manifest
from j_pretrain.data.shards import PackedDataset
from j_pretrain.data.tokenizer import tokenizer_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_CFG = REPO_ROOT / "configs" / "data" / "datasets.json"
DEFAULT_ARTIFACT_ROOT = "/home/hshi-j-4090/Desktop/j-pretrain-artifacts"

# probe source dir on disk (val split) -> corpus label in the probe tree
CORPORA = {"c4": "c4", "musicpile": "musicpile", "chempile": "chempile"}


def artifact_root() -> Path:
    return Path(os.environ.get("J_PRETRAIN_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT))


def build(root: Path | None = None) -> dict:
    cfg = json.loads(DATASETS_CFG.read_text())
    n_windows = int(cfg["probes"]["n_windows_per_corpus"])
    root = root or artifact_root()
    datasets_root = root / "datasets"
    probes_root = root / "probes"
    manifest_path = probes_root / "probe_manifest.json"
    if manifest_path.exists():
        print(f"[skip] probe manifest already exists: {manifest_path}")
        return json.loads(manifest_path.read_text())

    entries: dict[str, dict] = {}
    for disk_name, label in CORPORA.items():
        val_dir = datasets_root / disk_name / "val"
        val = PackedDataset(val_dir)
        rev = val.manifest.get("revision")
        probe_dir = probes_root / label
        m = build_probe(
            val, n_windows, probe_dir,
            meta={"source": label, "split": "val", "revision": rev,
                  "tokenizer_sha256": tokenizer_sha256(),
                  "source_val_manifest_n_seqs": val.manifest.get("n_seqs"),
                  "provenance": "first N val windows (frozen prefix); same probes for every ckpt"},
        )
        entries[label] = {
            "path": str(probe_dir.relative_to(root)),
            "n_windows": n_windows, "seq_len": m["seq_len"], "revision": rev,
            "tokenizer_sha256": tokenizer_sha256(),
            "shard_sha256": [s["sha256"] for s in m["shards"]],
        }
        print(f"[done] probe {label}: {n_windows} windows from {val_dir}")

    write_probe_manifest(probes_root, entries)
    print(f"[build_probes] wrote {manifest_path} ({len(entries)} corpora)")
    return entries


def main() -> None:
    build()


if __name__ == "__main__":
    main()
