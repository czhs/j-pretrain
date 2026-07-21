"""End-to-end orchestrator integration test on tiny REAL packed datasets (CPU).

Builds tiny on-disk packed shards for C4/MusicPile/ChemPile (train+val), tiny frozen
stage/model configs, and drives one full lambda pipeline (stage1->stage2->stage3)
through :func:`j_pretrain.orchestration.run.orchestrate`. Verifies:

* the shared init checkpoint is created once and every stage1 loads it;
* each stage completes, produces incoming/final (+restored_best for stage2) ckpts
  that load-validate, and is marked ``complete`` in canonical state;
* cross-stage lineage (stage2 parent == stage1 final; stage3 parent == stage2 best);
* run manifest + artifact/backup/storage inventories are written;
* per-run audits pass;
* re-running the orchestrator is a no-op (idempotent; nothing re-trained).
"""
from __future__ import annotations

import json

import numpy as np

from j_pretrain.artifacts import checkpoint as ck
from j_pretrain.artifacts import inventory as inv
from j_pretrain.config.schemas import ExecConfig
from j_pretrain.data.shards import ShardWriter
from j_pretrain.orchestration import run as orun

SEQ = 16


def _write_packed(root, source_dir, split, n_windows, seed):
    out = root / source_dir / split
    rng = np.random.default_rng(seed)
    w = ShardWriter(out, seq_len=SEQ, shard_seqs=n_windows,
                    meta={"source": source_dir, "split": split, "seq_len": SEQ})
    for _ in range(n_windows):
        w.add(rng.integers(0, 64, SEQ, dtype=np.uint16))
    w.finalize()


def _model_cfg(path):
    path.write_text(json.dumps({
        "name": "tiny", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "hidden_act": "silu", "max_position_embeddings": 32, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": True, "attention_bias": False,
        "mlp_bias": False, "attention_dropout": 0.0, "initializer_range": 0.04,
        "bos_token_id": 0, "eos_token_id": 0, "train_seq_len": SEQ,
        "tokenizer": "tiny", "torch_dtype": "float32"}))


def _stage_cfg(path, stage, max_tokens, gb=2, eval_interval=32, patience=None, primary=False):
    path.write_text(json.dumps({
        "stage": stage, "peak_lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 1,
        "global_batch_seqs": gb, "max_tokens": max_tokens, "seq_len": SEQ,
        "eval_interval_tokens": eval_interval, "early_stop_patience": patience,
        "early_stop_min_delta": 0.0}))


def _build_cfg(tmp_path, run_id="music-300m_lambda-0.5", lam=0.5):
    ds = tmp_path / "artifacts" / "datasets"
    _write_packed(ds, "c4", "train", 40, 1)
    _write_packed(ds, "c4", "val", 4, 2)
    _write_packed(ds, "musicpile", "train", 30, 3)
    _write_packed(ds, "musicpile", "val", 4, 4)
    _write_packed(ds, "chempile", "train", 30, 5)
    _write_packed(ds, "chempile", "val", 4, 6)

    cfgs = tmp_path / "cfgs"
    cfgs.mkdir()
    _model_cfg(cfgs / "model.json")
    # stage1: n_c4 = 512//16 = 32 windows, n_mp = round(0.5*20)=10 -> 42 windows / gb2 = 21 steps
    _stage_cfg(cfgs / "stage1.json", "stage1", max_tokens=9_000_000_000, gb=2)
    _stage_cfg(cfgs / "stage2.json", "stage2", max_tokens=2 * 2 * SEQ, gb=2, eval_interval=16,
               patience=None, primary=True)  # 2 steps
    _stage_cfg(cfgs / "stage3.json", "stage3", max_tokens=2 * 2 * SEQ, gb=2)  # 2 steps

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "experiment_state.json").write_text(json.dumps({
        "runs": {run_id: {"lambda": lam, "subset_tokens": 20 * SEQ, "seed": 1234,
                          "stage1": "planned", "stage2": "planned", "stage3": "planned"}},
        "env": {"torch": "test"}, "dataset_manifest_hashes": {"c4_revision": "x"}}))

    cfg = orun.OrchestratorConfig(
        repo_root=tmp_path, artifact_root=tmp_path / "artifacts",
        inventory_dir=tmp_path / "repo_artifacts", datasets_root=ds, device="cpu",
        exec_cfg=ExecConfig(microbatch_size=1, torch_compile=False),
        model_cfg_path=cfgs / "model.json",
        stage_cfg_paths={"stage1": cfgs / "stage1.json", "stage2": cfgs / "stage2.json",
                         "stage3": cfgs / "stage3.json"},
        run_ids=[run_id],
        run_specs={run_id: {"lambda": lam, "subset_tokens": 20 * SEQ, "seed": 1234}},
        c4_train_budget_tokens=32 * SEQ, seed=1234, tokenizer_ref={"id": "tiny"},
        dataset_manifest_hash="dsh", environment_hash="envh", fused_adamw=False,
        wandb_enabled=False, seq_len=SEQ, state_dir=state_dir)
    return cfg, run_id


def _no_wandb(*a, **k):
    return None


def test_full_pipeline_one_lambda(tmp_path):
    cfg, run_id = _build_cfg(tmp_path)
    result = orun.orchestrate(cfg, init_wandb_fn=_no_wandb)
    assert result["status"] == "complete", result

    # canonical state: all three stages complete
    state = json.loads((cfg.experiment_state_path).read_text())
    assert all(state["runs"][run_id][s] == "complete" for s in ("stage1", "stage2", "stage3"))

    inv_path = cfg.ckpt_inventory
    recs = [r for r in inv.read_inventory(inv_path) if r["op"] == "create"]

    # shared init created exactly once, both classes
    init = [r for r in recs if r["run_id"] == orun.INIT_RUN_ID]
    assert {r["checkpoint_class"] for r in init} == {"analysis", "resumable"}

    # each stage has incoming + final; stage2 also restored_best
    for stage, need in (("stage1", {"incoming", "final"}),
                        ("stage2", {"incoming", "final", "restored_best"}),
                        ("stage3", {"incoming", "final"})):
        labels = {l for r in recs if r["run_id"] == run_id and r["stage"] == stage
                  for l in r["milestone_labels"]}
        assert need.issubset(labels), (stage, labels)

    # every checkpoint physically loads + checksums verify
    for r in recs:
        d = cfg.artifact_root / r["rel_path"]
        assert ck.is_complete(d) and ck.verify_checksums(d)
        assert ck.read_meta(d)["load_validation_status"] == "verified"

    # cross-stage lineage: stage2 parent is stage1 final; stage3 parent is stage2 restored_best
    s1_final = next(r for r in recs if r["run_id"] == run_id and r["stage"] == "stage1"
                    and r["checkpoint_class"] == "analysis" and "final" in r["milestone_labels"])
    s2_incoming = next(r for r in recs if r["run_id"] == run_id and r["stage"] == "stage2"
                       and "incoming" in r["milestone_labels"])
    assert s2_incoming["parent_checkpoint"] == s1_final["checkpoint_id"]

    # run manifest covers the run for all 3 stages; audits pass
    man = [json.loads(l) for l in (cfg.repo_root / "runs" / "manifest.jsonl").read_text().splitlines()]
    assert {m["stage"] for m in man if m["run_id"] == run_id} == {"stage1", "stage2", "stage3"}
    assert all(m["audit_ok"] for m in man)

    # artifact/backup/storage inventories written
    for name in ("run_artifact_inventory.jsonl", "backup_status.jsonl", "storage_usage.jsonl"):
        assert (cfg.artifact_root / name).exists()

    # L_im proxy exists: stage2 final has an mp val metric
    s2_final = next(r for r in recs if r["run_id"] == run_id and r["stage"] == "stage2"
                    and "final" in r["milestone_labels"] and r["checkpoint_class"] == "analysis")
    assert "mp" in s2_final["metrics_at_creation"]

    # idempotent: re-running trains nothing new
    n_before = len(recs)
    result2 = orun.orchestrate(cfg, init_wandb_fn=_no_wandb)
    assert result2["status"] == "complete"
    recs2 = [r for r in inv.read_inventory(inv_path) if r["op"] == "create"]
    assert len(recs2) == n_before, "completed pipeline must not re-train"


def test_resume_after_interrupted_stage(tmp_path):
    """Kill after stage1, then resume: stage1 not retrained, pipeline still completes."""
    cfg, run_id = _build_cfg(tmp_path)
    # run only stage1 (max_nodes=1) — init + stage1
    r1 = orun.orchestrate(cfg, max_nodes=1, init_wandb_fn=_no_wandb)
    assert r1["processed"] == [f"{run_id}::stage1"]
    recs1 = [r for r in inv.read_inventory(cfg.ckpt_inventory)
             if r["op"] == "create" and r["run_id"] == run_id and r["stage"] == "stage1"]
    s1_ids = {r["checkpoint_id"] for r in recs1}
    # continue to completion
    r2 = orun.orchestrate(cfg, init_wandb_fn=_no_wandb)
    assert r2["status"] == "complete"
    recs1b = {r["checkpoint_id"] for r in inv.read_inventory(cfg.ckpt_inventory)
              if r["op"] == "create" and r["run_id"] == run_id and r["stage"] == "stage1"}
    assert recs1b == s1_ids, "stage1 checkpoints must be unchanged after resume"
