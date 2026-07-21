"""Tests for atomic checkpoint writing, load-validation and append-only inventory."""
from __future__ import annotations

import json

import pytest
import torch

from j_pretrain.artifacts import checkpoint as ck
from j_pretrain.artifacts import inventory as inv


def _tiny_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))


def _meta(cls: str, cid: str = "stage1-tok100-step10-abcd1234", **kw) -> ck.CheckpointMeta:
    base = dict(
        run_id="music-300m_lambda-0.25", checkpoint_id=cid, stage="stage1",
        checkpoint_class=cls, milestone_labels=["init"], lambda_frac=0.25,
        subset_tokens=300_000_000, step=10, total_tokens=100, c4_tokens=100,
        musicpile_tokens=0, chempile_tokens=0, lr=1e-3, train_loss=2.5,
        val_metrics={"c4_val": 3.1}, parent_checkpoint_id=None, config_hash="cfg",
        dataset_manifest_hash="dsh", environment_hash="envh", git_commit="deadbeef",
        seed=0, created_at_utc="2026-07-21T00:00:00Z",
        tokenizer_ref={"id": "SmolLM2-135M", "sha": "b4ec3f78"},
    )
    base.update(kw)
    return ck.CheckpointMeta(**base)


def test_write_analysis_snapshot(tmp_path):
    m = _tiny_model()
    final = ck.write_checkpoint(tmp_path, _meta(ck.CLASS_ANALYSIS), m, {"arch": "tiny"})
    assert final.exists() and ck.is_complete(final)
    assert (final / ck.WEIGHTS_NAME).exists()
    assert not (final / ck.TRAIN_STATE_NAME).exists()  # analysis has no optimizer state
    assert ck.verify_checksums(final)
    assert ck.read_meta(final)["load_validation_status"] == "verified"
    # weights reload and match
    sd = ck.load_weights(final)
    ref = m.state_dict()
    for k in ref:
        assert torch.allclose(sd[k].float(), ref[k].to(torch.bfloat16).float())


def test_write_resumable_roundtrip(tmp_path):
    m = _tiny_model()
    tstate = {"optimizer": {"lr": 1e-3}, "step": 10, "rng": {"a": 1}, "cursor": 42}
    final = ck.write_checkpoint(tmp_path, _meta(ck.CLASS_RESUMABLE), m, {"arch": "tiny"},
                                training_state=tstate)
    assert (final / ck.TRAIN_STATE_NAME).exists()
    got = ck.load_training_state(final)
    assert got["cursor"] == 42 and got["step"] == 10 and got["rng"] == {"a": 1}


def test_refuse_overwrite_final_path(tmp_path):
    m = _tiny_model()
    ck.write_checkpoint(tmp_path, _meta(ck.CLASS_ANALYSIS), m, {"arch": "tiny"})
    with pytest.raises(FileExistsError):
        ck.write_checkpoint(tmp_path, _meta(ck.CLASS_ANALYSIS), m, {"arch": "tiny"})


def test_no_tmp_dir_left_behind(tmp_path):
    m = _tiny_model()
    final = ck.write_checkpoint(tmp_path, _meta(ck.CLASS_ANALYSIS), m, {"arch": "tiny"})
    leftovers = list(final.parent.glob("*.tmp"))
    assert leftovers == []


def test_checksum_detects_corruption(tmp_path):
    m = _tiny_model()
    final = ck.write_checkpoint(tmp_path, _meta(ck.CLASS_ANALYSIS), m, {"arch": "tiny"})
    w = final / ck.WEIGHTS_NAME
    b = bytearray(w.read_bytes())
    b[-1] ^= 0xFF
    w.write_bytes(bytes(b))
    assert ck.verify_checksums(final) is False


def test_is_complete_ignores_incomplete_dir(tmp_path):
    d = tmp_path / "half"
    d.mkdir()
    (d / ck.WEIGHTS_NAME).write_bytes(b"x")
    (d / ck.META_NAME).write_text("{}")
    assert ck.is_complete(d) is False  # missing .complete marker


def test_inventory_append_and_live(tmp_path):
    path = tmp_path / "checkpoint_inventory.jsonl"
    m = _tiny_model()
    meta = _meta(ck.CLASS_RESUMABLE, cid="ck-1")
    ck.write_checkpoint(tmp_path, meta, m, {"arch": "tiny"}, training_state={"step": 1})
    inv.record_checkpoint(meta, rel_path="music/stage1/resumable/ck-1", byte_size=123,
                          created_at_utc="2026-07-21T00:00:00Z", inventory_path=path)
    recs = inv.read_inventory(path)
    assert len(recs) == 1 and recs[0]["op"] == "create" and recs[0]["checkpoint_id"] == "ck-1"
    assert recs[0]["backup_status"] == "unreplicated_local_copy"
    assert inv.live_checkpoints(path) == {"ck-1"}


def test_inventory_prune_supersedes(tmp_path):
    path = tmp_path / "checkpoint_inventory.jsonl"
    meta = _meta(ck.CLASS_RESUMABLE, cid="ck-old")
    inv.record_checkpoint(meta, "p", 100, "2026-07-21T00:00:00Z", inventory_path=path)
    inv.record_checkpoint(_meta(ck.CLASS_RESUMABLE, cid="ck-new"), "p2", 100,
                          "2026-07-21T01:00:00Z", inventory_path=path)
    inv.record_prune("music-300m_lambda-0.25", "ck-old", reason="superseded",
                     superseded_by="ck-new", freed_bytes=100, at_utc="2026-07-21T02:00:00Z",
                     inventory_path=path)
    # append-only: 3 lines, original create still present
    assert len(inv.read_inventory(path)) == 3
    assert inv.live_checkpoints(path) == {"ck-new"}  # ck-old pruned, not live
