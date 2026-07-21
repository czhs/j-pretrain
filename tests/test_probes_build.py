"""Probe-builder driver test: fixed-prefix probes from tiny val packed datasets."""
from __future__ import annotations

import json

import numpy as np

from j_pretrain.data.shards import PackedDataset, ShardWriter
from j_pretrain.data.tokenizer import tokenizer_sha256

import scripts.build_probes as bp  # noqa: E402

SEQ = 16
N_PROBE = json.loads((bp.DATASETS_CFG).read_text())["probes"]["n_windows_per_corpus"]


def _val(root, name, n, seed, rev):
    out = root / "datasets" / name / "val"
    rng = np.random.default_rng(seed)
    w = ShardWriter(out, seq_len=SEQ, shard_seqs=n, meta={"source": name, "split": "val",
                    "seq_len": SEQ, "revision": rev})
    for _ in range(n):
        w.add(rng.integers(0, 64, SEQ, dtype=np.uint16))
    w.finalize()


def test_build_probes_prefix_and_manifest(tmp_path):
    _val(tmp_path, "c4", N_PROBE + 4, 1, "c4rev")
    _val(tmp_path, "musicpile", N_PROBE + 4, 2, "mprev")
    _val(tmp_path, "chempile", N_PROBE + 4, 3, "chrev")

    entries = bp.build(root=tmp_path)
    man = json.loads((tmp_path / "probes" / "probe_manifest.json").read_text())
    assert set(man.keys()) == {"c4", "musicpile", "chempile"} == set(entries.keys())

    for label, rev in (("c4", "c4rev"), ("musicpile", "mprev"), ("chempile", "chrev")):
        probe = PackedDataset(tmp_path / "probes" / label)
        assert len(probe) == N_PROBE
        assert probe.verify_checksums()
        assert man[label]["revision"] == rev
        assert man[label]["tokenizer_sha256"] == tokenizer_sha256()
        # probe is the exact prefix of the source val ordering
        src = PackedDataset(tmp_path / "datasets" / label / "val")
        assert np.array_equal(probe[0], src[0]) and np.array_equal(probe[N_PROBE - 1], src[N_PROBE - 1])

    # idempotent: second call is a no-op skip (manifest unchanged)
    before = (tmp_path / "probes" / "probe_manifest.json").read_text()
    bp.build(root=tmp_path)
    assert (tmp_path / "probes" / "probe_manifest.json").read_text() == before
