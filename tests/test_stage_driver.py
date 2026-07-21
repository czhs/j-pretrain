"""End-to-end StageDriver test on a tiny model + fake packed datasets:
milestone checkpoints (incoming/scheduled/final), eval logging, inventory records,
checkpoint load-validation, and stage-2 best/early-stopping.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from j_pretrain.artifacts import checkpoint as ck
from j_pretrain.artifacts import inventory as inv
from j_pretrain.config.schemas import ExecConfig, StageConfig
from j_pretrain.orchestration.metrics import MetricLogger
from j_pretrain.orchestration.stage_driver import StageContext, StageDriver
from j_pretrain.training.dataplan import ShuffledSourcePlan
from j_pretrain.training.loader import PlanLoader
from j_pretrain.training.loop import Trainer


class FakePacked:
    def __init__(self, arr):
        self.arr = arr
        self.seq_len = arr.shape[1]
    def __len__(self):
        return self.arr.shape[0]
    def __getitem__(self, i):
        return self.arr[i]


def _tiny_lm(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=32, rms_norm_eps=1e-5,
                      tie_word_embeddings=True, attention_bias=False, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg)


def _ctx(stage, max_tokens, parent=None):
    return StageContext(
        run_id="music-300m_lambda-0.5", stage=stage, lambda_frac=0.5,
        subset_tokens=300_000_000, seed=1234, milestone_max_tokens=max_tokens,
        config_hash="cfg", dataset_manifest_hash="dsh", environment_hash="env",
        git_commit="abc123", tokenizer_ref={"id": "SmolLM2"}, parent_checkpoint_id=parent)


def _driver(tmp_path, stage, source, total_steps, a_marks, r_marks, primary_val=None,
            eval_interval=48, es_patience=None, seed=0):
    seq = 16
    pool = np.random.default_rng(1).integers(0, 64, (40, seq), dtype=np.uint16)
    ds = FakePacked(pool)
    plan = ShuffledSourcePlan(source, len(ds), seed=2)
    loader = PlanLoader(plan, {source: ds}, seq_len=seq)
    cfg = StageConfig(stage=stage, peak_lr=1e-3, min_lr=1e-4, warmup_steps=1,
                      global_batch_seqs=2, max_tokens=total_steps * 2 * seq, seq_len=seq,
                      eval_interval_tokens=eval_interval, early_stop_patience=es_patience)
    tr = Trainer(_tiny_lm(seed), cfg, ExecConfig(microbatch_size=1, torch_compile=False),
                 loader, total_steps=total_steps, device="cpu")
    val = FakePacked(np.random.default_rng(9).integers(0, 64, (4, seq), dtype=np.uint16))
    logger = MetricLogger("music-300m_lambda-0.5", stage,
                          tmp_path / "metrics.jsonl", wandb_run=None)
    clock = {"n": 0}
    def now():
        clock["n"] += 1
        return f"2026-07-21T00:00:{clock['n']:02d}Z"
    drv = StageDriver(tr, _ctx(stage, cfg.max_tokens), cfg, {"arch": "tiny"},
                      artifact_root=tmp_path, val_sets={source: val},
                      logger=logger, now_fn=now, analysis_token_marks=a_marks,
                      resumable_token_marks=r_marks, primary_val=primary_val)
    return drv, tr, tmp_path


def test_stage_driver_writes_milestones_and_inventory(tmp_path):
    # 6 steps * 2 seqs * 16 tok = 192 driver tokens; marks reachable in-test
    drv, tr, root = _driver(tmp_path, "stage3", "chempile", total_steps=6,
                            a_marks=[32, 96], r_marks=[64])
    result = drv.run()
    assert result["opt_step"] == 6
    assert result["driver_tokens"] == 192 and result["tokens"]["chempile"] == 192

    recs = inv.read_inventory(root / inv.CHECKPOINT_INVENTORY)
    labels = [tuple(r["milestone_labels"]) for r in recs if r["op"] == "create"]
    classes = {(r["checkpoint_class"], tuple(r["milestone_labels"])) for r in recs}
    # incoming + final each have BOTH analysis and resumable
    assert ("analysis", ("incoming",)) in classes and ("resumable", ("incoming",)) in classes
    assert ("analysis", ("final",)) in classes and ("resumable", ("final",)) in classes
    # scheduled analysis snapshot(s) at 32 and/or 96, and a scheduled resumable at 64
    assert any(l[0].startswith("tok") for l in labels)
    assert ("scheduled",) in labels

    # every recorded checkpoint physically exists, load-validates, checksums verify
    for r in recs:
        if r["op"] != "create":
            continue
        d = root / r["rel_path"]
        assert ck.is_complete(d)
        assert ck.read_meta(d)["load_validation_status"] == "verified"
        assert ck.verify_checksums(d)

    # metrics log has incoming, eval, final events
    events = [__import__("json").loads(l)["event"]
              for l in (root / "metrics.jsonl").read_text().splitlines()]
    assert "incoming" in events and "final" in events and "eval" in events


def test_stage_driver_unique_ids_no_overwrite(tmp_path):
    drv, tr, root = _driver(tmp_path, "stage3", "chempile", total_steps=4,
                            a_marks=[32], r_marks=[64])
    drv.run()
    recs = [r for r in inv.read_inventory(root / inv.CHECKPOINT_INVENTORY)
            if r["op"] == "create"]
    ids = [r["checkpoint_id"] for r in recs]
    assert len(ids) == len(set(ids)), "checkpoint ids must be unique (no overwrite)"


def test_inventory_dir_decoupled_from_payload_root(tmp_path):
    """Inventory goes to inventory_dir (committed repo/artifacts); payloads to artifact_root."""
    payload_root = tmp_path / "external"
    inv_dir = tmp_path / "repo_artifacts"
    inv_dir.mkdir()
    drv, tr, _ = _driver(payload_root, "stage3", "chempile", total_steps=2,
                         a_marks=[], r_marks=[])
    drv.inventory_dir = inv_dir  # simulate orchestrator wiring
    drv.run()
    # inventory written under inv_dir, not payload_root
    assert (inv_dir / inv.CHECKPOINT_INVENTORY).exists()
    assert not (payload_root / inv.CHECKPOINT_INVENTORY).exists()
    recs = [r for r in inv.read_inventory(inv_dir / inv.CHECKPOINT_INVENTORY)
            if r["op"] == "create"]
    assert recs
    # rel_path locates the checkpoint under payload_root
    for r in recs:
        assert (payload_root / r["rel_path"]).exists()


def test_stage2_best_and_early_stop(tmp_path):
    # primary_val set + tiny patience: after best plateaus, early-stop fires
    drv, tr, root = _driver(tmp_path, "stage2", "mp", total_steps=20,
                            a_marks=[], r_marks=[], primary_val="mp",
                            eval_interval=32, es_patience=1)
    result = drv.run()
    recs = inv.read_inventory(root / inv.CHECKPOINT_INVENTORY)
    classes = {(r["checkpoint_class"], tuple(r["milestone_labels"]))
               for r in recs if r["op"] == "create"}
    assert ("analysis", ("best",)) in classes  # a best-val checkpoint was saved
    assert ("analysis", ("restored_best",)) in classes  # restored best for stage 3
    assert result["best_val"] is not None
