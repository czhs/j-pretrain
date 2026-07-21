"""Unit tests for the deterministic data pipeline (no network / heavy corpora)."""
from __future__ import annotations

import numpy as np
import pytest

from j_pretrain.data.interleave import (
    interleave_schedule,
    lambda_plan,
    schedule_source_indices,
)
from j_pretrain.data.packing import count_packed_windows, pack_documents
from j_pretrain.data.probes import build_probe
from j_pretrain.data.shards import PackedDataset, ShardWriter
from j_pretrain.data.splits import SPLIT_TRAIN, SPLIT_VAL, SplitPolicy, doc_key, split_of
from j_pretrain.data.subset import SubsetSpec, is_nested, subset_window_range

SEQ = 8


# ---------------------------------------------------------------- packing
def test_packing_exact_counts_and_boundaries():
    docs = [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12]]  # 12 tokens total
    st = count_packed_windows(iter(docs), seq_len=4)
    assert st.n_documents == 3
    assert st.n_tokens_consumed == 12
    assert st.n_windows == 3
    assert st.n_tokens_emitted == 12
    assert st.n_tokens_dropped == 0


def test_packing_drops_remainder_and_counts_it():
    docs = [[1, 2, 3, 4, 5]]  # 5 tokens, seq_len 4 -> 1 window, 1 dropped
    windows = list(pack_documents(iter(docs), seq_len=4))
    assert len(windows) == 1
    assert windows[0].tolist() == [1, 2, 3, 4]
    st = count_packed_windows(iter(docs), seq_len=4)
    assert st.n_windows == 1 and st.n_tokens_dropped == 1


def test_packing_deterministic():
    docs = [[i] * 3 for i in range(1, 20)]
    a = [w.tolist() for w in pack_documents(iter(docs), SEQ)]
    b = [w.tolist() for w in pack_documents(iter(docs), SEQ)]
    assert a == b


def test_packing_no_window_overlap():
    docs = [list(range(1, 100))]
    windows = list(pack_documents(iter(docs), SEQ))
    flat = np.concatenate(windows).tolist()
    # windows are consecutive, non-overlapping slices of the stream
    assert flat == list(range(1, 1 + len(flat)))


# ---------------------------------------------------------------- shards
def _make_packed(tmp_path, n_seqs, seq_len=SEQ, meta=None):
    w = ShardWriter(tmp_path, seq_len=seq_len, shard_seqs=3, meta=meta or {})
    for i in range(n_seqs):
        w.add(np.arange(i * seq_len, (i + 1) * seq_len, dtype=np.uint16))
    w.finalize()
    return PackedDataset(tmp_path)


def test_shard_roundtrip_and_len(tmp_path):
    ds = _make_packed(tmp_path, 7)
    assert len(ds) == 7
    assert ds[0].tolist() == list(range(0, SEQ))
    assert ds[6].tolist() == list(range(48, 56))
    assert ds[-1].tolist() == ds[6].tolist()


def test_shard_checksums_verify(tmp_path):
    ds = _make_packed(tmp_path, 5)
    assert ds.verify_checksums() is True


def test_shard_checksum_detects_corruption(tmp_path):
    ds = _make_packed(tmp_path, 5)
    shard = tmp_path / ds.manifest["shards"][0]["name"]
    b = bytearray(shard.read_bytes())
    b[-1] ^= 0xFF
    shard.write_bytes(bytes(b))
    assert ds.verify_checksums() is False


def test_shard_manifest_token_count(tmp_path):
    ds = _make_packed(tmp_path, 6)
    assert ds.manifest["n_tokens"] == 6 * SEQ


# ---------------------------------------------------------------- splits
def test_split_deterministic_and_disjoint():
    pol = SplitPolicy(val_pm=100, tune_pm=100, salt="c4")
    keys = [doc_key("c4", "rev", str(i)) for i in range(5000)]
    a = {k: split_of(k, pol) for k in keys}
    b = {k: split_of(k, pol) for k in keys}
    assert a == b
    for k in keys:  # each key in exactly one split
        assert a[k] in {"train", "tune", "val"}


def test_split_ratios_approximately_hit_bands():
    pol = SplitPolicy(val_pm=50, tune_pm=50, salt="mp")
    keys = [doc_key("mp", "rev", str(i)) for i in range(20000)]
    counts = {"train": 0, "tune": 0, "val": 0}
    for k in keys:
        counts[split_of(k, pol)] += 1
    assert 0.03 < counts["val"] / len(keys) < 0.07
    assert 0.03 < counts["tune"] / len(keys) < 0.07
    assert counts["train"] + counts["tune"] + counts["val"] == len(keys)


def test_split_salt_changes_assignment():
    k = doc_key("x", "rev", "42")
    assert split_of(k, SplitPolicy(salt="a")) or True  # deterministic call
    # different salts should not produce identical global mapping
    keys = [doc_key("x", "rev", str(i)) for i in range(2000)]
    ma = [split_of(k, SplitPolicy(val_pm=200, salt="a")) for k in keys]
    mb = [split_of(k, SplitPolicy(val_pm=200, salt="b")) for k in keys]
    assert ma != mb


# ---------------------------------------------------------------- subset
def test_subset_exact_token_count():
    spec = SubsetSpec(target_tokens=300_000_000, seq_len=1024)
    assert spec.n_windows == 300_000_000 // 1024
    assert spec.exact_tokens == spec.n_windows * 1024
    assert spec.exact_tokens <= 300_000_000


def test_subset_window_range_prefix():
    r = subset_window_range(300_000_000, 1024, pool_windows=400_000)
    assert r == range(0, 300_000_000 // 1024)


def test_subset_pool_too_small_raises():
    with pytest.raises(ValueError):
        subset_window_range(300_000_000, 1024, pool_windows=10)


def test_subset_nested():
    assert is_nested(30_000_000, 150_000_000, 1024)
    assert is_nested(150_000_000, 300_000_000, 1024)
    # prefix property: smaller range is a prefix of larger
    small = subset_window_range(30_000_000, 1024, 400_000)
    large = subset_window_range(300_000_000, 1024, 400_000)
    assert list(small) == list(large)[: len(small)]


# ---------------------------------------------------------------- lambda / interleave
@pytest.mark.parametrize("lam,expected_mp_tokens", [
    (0.0, 0),
    (0.25, (round(0.25 * (300_000_000 // 1024))) * 1024),
    (0.5, (round(0.5 * (300_000_000 // 1024))) * 1024),
    (0.75, (round(0.75 * (300_000_000 // 1024))) * 1024),
    (1.0, (300_000_000 // 1024) * 1024),
])
def test_lambda_token_allocation(lam, expected_mp_tokens):
    p = lambda_plan(lam, c4_budget_tokens=8_700_000_000, subset_tokens=300_000_000, seq_len=1024)
    assert p.c4_tokens == (8_700_000_000 // 1024) * 1024  # fixed across lambda
    assert p.mp_tokens == expected_mp_tokens


def test_lambda_c4_budget_fixed_across_lambda():
    plans = [lambda_plan(l, 8_700_000_000, 300_000_000, 1024) for l in (0, 0.25, 0.5, 0.75, 1.0)]
    assert len({p.n_c4_windows for p in plans}) == 1  # identical C4 count


def test_interleave_exact_source_counts():
    for n_c4, n_mp in [(100, 0), (100, 25), (97, 13), (8_496_093, 292_968)]:
        tags = interleave_schedule(n_c4, n_mp) if n_c4 < 1000 else None
        if tags is not None:
            tl = list(tags)
            assert tl.count("c4") == n_c4
            assert tl.count("mp") == n_mp
        else:
            # large: count via streaming without materialising huge list
            c = m = 0
            for t in interleave_schedule(n_c4, n_mp):
                if t == "c4":
                    c += 1
                else:
                    m += 1
            assert c == n_c4 and m == n_mp


def test_interleave_uniform_not_frontloaded():
    n_c4, n_mp = 90, 10
    tags = list(interleave_schedule(n_c4, n_mp))
    mp_pos = [i for i, t in enumerate(tags) if t == "mp"]
    # 10 mp centered across 100 slots -> spacing 10, half-band (~5) at each end
    assert max(mp_pos) < len(tags) - 1  # final window is C4, not MusicPile
    assert min(mp_pos) > 0              # first window is C4, not MusicPile
    gaps = np.diff(mp_pos)              # consecutive mp spacing is uniform
    assert gaps.max() - gaps.min() <= 1


def test_interleave_c4_order_invariant_across_lambda():
    # C4 window indices must appear in identical relative order regardless of n_mp
    def c4_order(n_mp):
        return [idx for src, idx in schedule_source_indices(50, n_mp) if src == "c4"]
    base = c4_order(0)
    for n_mp in (5, 17, 40):
        assert c4_order(n_mp) == base  # 0,1,2,...,49 preserved


def test_schedule_source_indices_consumes_prefixes():
    pairs = list(schedule_source_indices(6, 3))
    c4_idx = [i for s, i in pairs if s == "c4"]
    mp_idx = [i for s, i in pairs if s == "mp"]
    assert c4_idx == list(range(6))  # in-order prefix
    assert mp_idx == list(range(3))  # in-order prefix (nested subset)


# ---------------------------------------------------------------- probes
def test_probe_deterministic_prefix(tmp_path):
    val = _make_packed(tmp_path / "val", 20, meta={"source": "c4", "split": "val"})
    p1 = build_probe(val, 5, tmp_path / "probe1", meta={"source": "c4"})
    p2 = build_probe(val, 5, tmp_path / "probe2", meta={"source": "c4"})
    d1 = PackedDataset(tmp_path / "probe1")
    d2 = PackedDataset(tmp_path / "probe2")
    assert len(d1) == 5 == len(d2)
    for i in range(5):
        assert d1[i].tolist() == d2[i].tolist() == val[i].tolist()
    assert p1["shards"][0]["sha256"] == p2["shards"][0]["sha256"]


def test_probe_too_large_raises(tmp_path):
    val = _make_packed(tmp_path / "val", 3)
    with pytest.raises(ValueError):
        build_probe(val, 10, tmp_path / "probe", meta={})


# ---------------------------------------------------------------- no-leakage (structural)
def test_no_chempile_in_stage1_schedule():
    # Stage-1 schedule only ever references c4 / mp sources; chempile cannot appear.
    sources = {src for src, _ in schedule_source_indices(20, 5)}
    assert sources == {"c4", "mp"}


def test_stage1_mp_uses_train_subset_prefix_only():
    # MusicPile exposure indexes into the subset prefix [0, n_mp); it never draws
    # from val (subset is built from the train pool; val is a disjoint hash band).
    p = lambda_plan(0.5, 8_700_000_000, 300_000_000, 1024)
    mp_idx = [i for s, i in _stream_head(schedule_source_indices(p.n_c4_windows, p.n_mp_windows), 500) if s == "mp"]
    assert all(0 <= i < p.n_mp_windows for i in mp_idx)


def _stream_head(it, n):
    out = []
    for x in it:
        out.append(x)
        if len(out) >= n:
            break
    return out
