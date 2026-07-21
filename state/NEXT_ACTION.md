# NEXT ACTION

**Phase:** scope_locked (training+artifacts+stage-configs done; SCOPE_LOCK.json written; 67 tests pass)

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

**DONE since last:** stage configs frozen (configs/stage{1,2,3}); storage bench (analysis 310.6MB,
resumable 1850MB) -> STORAGE_PLAN.md; FEASIBILITY.md gate PASS; SCOPE_LOCK.json (scope_hash ca992877,
5 runs music-300m_lambda-{0.0..1.0}); run_queue + runs populated in experiment_state.

**Next exact action (fresh session):**
1. `cat state/experiment_state.json state/NEXT_ACTION.md state/SCOPE_LOCK.json`; `pytest tests/ -q` (67).
2. Build `src/j_pretrain/orchestration/`: (a) run DAG per RunSpec (stage1->2->3, lambda=0 stage1 feeds
   ONLY lambda=0), (b) exclusive GPU lock file (state/gpu.lock w/ pid+staleness detect), (c) tmux
   durable runner launching ONE stage at a time via env python, (d) state/run_queue.json + update
   process_registry.json, (e) resume-detection (skip validated artifacts; validate config+lineage
   hashes; never restart healthy run), (f) retry transient <=3 then failed_blocked. A stage driver
   that wires Trainer + checkpoint schedules (schedule.crossed_*) + eval + wandb (entity ametind-o,
   proj j-pretrain, run=run_id, group by stage/lambda) + inventory.record_checkpoint.
3. `scripts/verify_completion.py` — deterministic, machine-readable, nonzero on any unmet criterion
   (see mission "Completion verifier" list). Writes reports/completion_verification.json + summary.
4. Per-run completion audit fn (config hash match, lineage, token counts, ckpts load, metrics complete).
5. PRETRAIN_READINESS_AUDIT via fresh-context subagent -> reports/PRETRAIN_READINESS_AUDIT.md; resolve
   criticals BEFORE Stage 1.
6. When build_datasets.py EXIT_CODE=0: build probes (data/probes.py, first 256 val windows/corpus) +
   record dataset_manifest_hashes in experiment_state -> save permanent init ckpt (seed 1234, shared
   across all 5 stage1 runs) -> launch Stage 1 lambda=0 first, detached via orchestrator.

**Must NOT repeat:** preflight, env, spec, benchmark, config/model, data pipeline+tests, tokenizer
freeze, revision pin, DATA_AUDIT, training+artifacts modules+tests, resume test (all done).
**Must NOT:** commit weights/datasets/*.npy/logs/wandb runs; use `conda run`; reduce scope; train
before readiness audit + scope lock + feasibility gate; restart healthy build_datasets.

**Command to resume orchestrator:** none yet (orchestrator not built).
**Command to build datasets (full, already running):** `HF_HUB_ENABLE_HF_TRANSFER=0
TOKENIZERS_PARALLELISM=false J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts
/home/hshi-j-4090/miniconda3/envs/jpre/bin/python scripts/build_datasets.py`
