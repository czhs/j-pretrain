# NEXT ACTION

**Phase:** training_pipeline_built (trainer + atomic checkpoints + resume test done; 67 tests pass)

**Process running?** YES — `build_datasets_full` (CPU tokenization, tmux `dbuild`, pid 1100073,
NOT GPU). Log: `logs/build_datasets_20260721T133614Z.log`. Output:
`/home/hshi-j-4090/Desktop/j-pretrain-artifacts/datasets`. Resumable (skips complete manifests),
fails loudly if a pool is short of budget. ~11 min/shard (102.4M tok/shard); C4 alone ~85 shards
(~15h) + MP + ChemPile. DO NOT restart if healthy. Health-check ~10 min:
`tail -n 3 <log>; ps -p 1100073 -o etime=; ls <output>/*/*/ | wc -l`. On EXIT_CODE=0 -> all 6
(source,split) manifests exist; build probes + record dataset_manifest_hashes. On non-zero /
pool-short -> inspect; genuine short pool = candidate external BLOCKER.

**CRITICAL OPS NOTE:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY (NOT
`conda run -n jpre` — swallows heredoc stdout). Set `HF_HUB_ENABLE_HF_TRANSFER=0
TOKENIZERS_PARALLELISM=false` and `J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`
for HF/data ops.

**Last verified (2026-07-21):**
- 67 tests pass (`pytest tests/ -q`). NEW this iter: `src/j_pretrain/training/` (schedule, dataplan,
  loader, optim, rngstate, loop) + `src/j_pretrain/artifacts/` (checkpoint, inventory) + StageConfig
  in config/schemas.py. tests/test_training.py (18) + tests/test_artifacts.py (8).
- **Deterministic resume test passes** (train N vs train K->save->restart into fresh Trainer->N;
  losses match abs 1e-5, params allclose). Data order = pure fn of integer cursor (dataplan).
- Trainer: fp32 master params + bf16 autocast on CUDA (mixed precision, no GradScaler); AdamW
  b1=.9 b2=.95 clip1.0, cosine+warmup, exact per-source token counters, token-weighted eval CE.
- Atomic ckpt writer: analysis (bf16 safetensors) + resumable (fp32 model+opt+rng+cursor in
  training_state.pt); tmp->fsync->checksum->load-test->rename; append-only inventory.

**Next exact action (fresh session):**
1. `cat state/experiment_state.json state/NEXT_ACTION.md`; `pytest tests/ -q` (expect 67 pass).
2. Freeze stage configs: `configs/stage1/music.json`, `configs/stage2/music.json`,
   `configs/stage3/chempile.json` as StageConfig JSON (see schemas.StageConfig). Stage1/3 LOCAL
   (Stage1: peak 5e-4/min 5e-5/warmup ~1000/gbs 512/WD 0.1/max=per-lambda total; Stage3: lr 5e-5/
   200M/gbs match S2); Stage2 provisional (LR5e-4 db0 gbs480 warmup500 WD0.1 minLR5e-5 max2B ES p3)
   pending lambda=0 pilot. Record in DECISIONS.
3. Storage+ckpt-size bench (GPU free; build_datasets is CPU): build real 135M init, write 1 analysis
   + 1 resumable ckpt via artifacts.checkpoint.write_checkpoint, measure bytes -> docs/STORAGE_PLAN.md;
   finalize docs/COMPUTE_PLAN.md + reports/FEASIBILITY.md (21-day / storage gate; BLOCKED.md if fails).
4. Orchestration `src/j_pretrain/orchestration/`: run DAG, exclusive GPU lock, tmux durable runner,
   process_registry/run_queue.json, resume-detection, per-run completion audit. scripts/verify_completion.py.
5. PRETRAIN_READINESS_AUDIT (fresh-context subagent) -> resolve criticals -> state/SCOPE_LOCK.json.
6. When build_datasets.py EXIT_CODE=0: build probes (data/probes.py) + record dataset_manifest_hashes
   -> save permanent init ckpt -> launch Stage 1 (lambda=0 first) detached via orchestrator.

**Must NOT repeat:** preflight, env, spec, benchmark, config/model, data pipeline+tests, tokenizer
freeze, revision pin, DATA_AUDIT, training+artifacts modules+tests, resume test (all done).
**Must NOT:** commit weights/datasets/*.npy/logs/wandb runs; use `conda run`; reduce scope; train
before readiness audit + scope lock + feasibility gate; restart healthy build_datasets.

**Command to resume orchestrator:** none yet (orchestrator not built).
**Command to build datasets (full, already running):** `HF_HUB_ENABLE_HF_TRANSFER=0
TOKENIZERS_PARALLELISM=false J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts
/home/hshi-j-4090/miniconda3/envs/jpre/bin/python scripts/build_datasets.py`
