"""Tests for deterministic config hashing and run-id derivation."""
from j_pretrain.config.hashing import (
    canonical_json,
    config_hash,
    derive_run_id,
    short_hash,
)


def test_canonical_json_key_order_invariant():
    a = {"b": 1, "a": 2, "c": {"y": 1, "x": 2}}
    b = {"c": {"x": 2, "y": 1}, "a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_float_repr_stable():
    # 1e-5 and 0.00001 are the same float -> same hash.
    assert config_hash({"eps": 1e-5}) == config_hash({"eps": 0.00001})


def test_config_hash_changes_on_value_change():
    h1 = config_hash({"lr": 5e-4})
    h2 = config_hash({"lr": 1e-3})
    assert h1 != h2


def test_config_hash_deterministic():
    obj = {"lr": 5e-4, "layers": 30, "nested": [1, 2, {"k": "v"}]}
    assert config_hash(obj) == config_hash(obj)
    assert len(config_hash(obj)) == 64
    assert short_hash(obj, 8) == config_hash(obj)[:8]


def test_derive_run_id_format():
    assert derive_run_id("music", 300_000_000, 0.0) == "music-300m_lambda-0.0"
    assert derive_run_id("music", 300_000_000, 0.25) == "music-300m_lambda-0.25"
    assert derive_run_id("music", 300_000_000, 1.0) == "music-300m_lambda-1.0"
    assert derive_run_id("music", 150_000_000, 0.5) == "music-150m_lambda-0.5"


def test_run_ids_unique_across_grid():
    ids = {derive_run_id("music", 300_000_000, l) for l in (0.0, 0.25, 0.5, 0.75, 1.0)}
    assert len(ids) == 5
