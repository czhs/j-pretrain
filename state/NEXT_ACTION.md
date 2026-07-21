# NEXT ACTION

**Phase:** orchestration_core_built (gpu-lock + DAG + stage-driver + metrics done; 82 tests pass)

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

**DONE since last:** orchestration core — `orchestration/gpulock.py` (exclusive GPU lock, stale
reclaim), `orchestration/dag.py` (run DAG, next_node), `orchestration/stage_driver.py` (StageDriver:
milestone ckpts + eval + best/early-stop + inventory), `orchestration/metrics.py` (local JSONL +
wandb). tests/test_orchestration.py (12) + tests/test_stage_driver.py (3). Stage configs, storage
bench, FEASIBILITY gate PASS, SCOPE_LOCK (ca992877, 5 runs) all done in prior sub-iters.

**Next exact action (fresh session):**
1. `cat state/experiment_state.json state/NEXT_ACTION.md state/SCOPE_LOCK.json`; `pytest tests/ -q` (82).
2. Build orchestrator entrypoint `src/j_pretrain/orchestration/run.py` (`python -m j_pretrain.orchestration.run`):
   read experiment_state+SCOPE_LOCK, `dag.next_node` (order=run_queue, lambda=0 first), acquire
   GpuLock(state/gpu.lock), build REAL sources: stage1={c4-train, mp-train(subset prefix)} via
   PackedDataset + Stage1Plan(lambda_plan windows); stage2=mp-train subset via ShuffledSourcePlan;
   stage3=chempile-train via ShuffledSourcePlan; val_sets={c4,mp,chempile}-val. Build model, load
   parent weights (stage1<-init ckpt; stage2<-stage1 final; stage3<-stage2 restored_best). Run
   StageDriver. Update experiment_state runs[*][stage] + process_registry + run_queue.json. Detached
   via tmux; resume-detect (load latest valid resumable; never restart healthy). retry<=3.
3. `scripts/verify_completion.py` — full mandatory-criteria checklist -> reports/completion_verification.json,
   nonzero on any unmet. (See mission "Completion verifier".)
4. Per-run completion audit fn (config-hash match, lineage, token counts, ckpts load, metrics complete).
5. PRETRAIN_READINESS_AUDIT via fresh-context subagent -> reports/PRETRAIN_READINESS_AUDIT.md; resolve
   criticals BEFORE Stage 1.
6. When build_datasets.py EXIT_CODE=0: build probes (data/probes.py) + record dataset_manifest_hashes
   -> save shared init ckpt (seed 1234) -> launch Stage 1 lambda=0 first, detached via orchestrator.

**Key facts for entrypoint:** run_ids = music-300m_lambda-{0.0,0.25,0.5,0.75,1.0}; seed 1234;
Stage1 total_steps per-lambda = (8496093 + round(lambda*292968))//512; Stage2/3 total_steps =
StageConfig.max_optimizer_steps. Stage2 uses FULL 292968-window subset every lambda. artifact_root =
/home/hshi-j-4090/Desktop/j-pretrain-artifacts; ckpts under <root>/checkpoints/<run>/<stage>/<class>/.

**Must NOT repeat:** preflight, env, spec, benchmark, config/model, data pipeline+tests, tokenizer
freeze, revision pin, DATA_AUDIT, training+artifacts modules+tests, resume test (all done).
**Must NOT:** commit weights/datasets/*.npy/logs/wandb runs; use `conda run`; reduce scope; train
before readiness audit + scope lock + feasibility gate; restart healthy build_datasets.

**Command to resume orchestrator:** none yet (orchestrator not built).
**Command to build datasets (full, already running):** `HF_HUB_ENABLE_HF_TRANSFER=0
TOKENIZERS_PARALLELISM=false J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts
/home/hshi-j-4090/miniconda3/envs/jpre/bin/python scripts/build_datasets.py`
