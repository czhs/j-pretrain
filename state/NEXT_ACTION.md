# NEXT ACTION

**Phase:** data_pipeline_built (data modules + 29 tests + smoke-validated driver; tokenizer frozen)

**Process running?** No training/GPU job. No detached runner. Nothing to babysit.

**CRITICAL OPS NOTE:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY (NOT
`conda run -n jpre` — swallows heredoc stdout). Set `HF_HUB_ENABLE_HF_TRANSFER=0
TOKENIZERS_PARALLELISM=false` and `J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`
for HF/data ops.

**Last verified (2026-07-21):**
- 41 tests pass (`pytest tests/ -q`). Data pipeline complete: tokenizer.py, packing.py, shards.py,
  splits.py, subset.py, interleave.py, probes.py. Tokenizer frozen -> configs/data/tokenizer/ (4.6M).
  Revisions pinned in configs/data/datasets.json. docs/DATA_AUDIT.md written.
- Smoke build (scripts/build_datasets.py --smoke) built all 6 (source,split) packed sets OK;
  checksums verify; mmap load OK. Cleaned up.
- Verified real numbers: 300M subset = 292,968 win = 299,999,232 tok. C4 fixed 8,496,093 win
  (~8.7B) across ALL lambda. MP tokens: 0/75M/150M/225M/300M exactly. (table in DATA_AUDIT.md)
- Commits so far: 3b40459 (spec), d4a1bd4 (config+model), + this data-pipeline commit.

**Next exact action (fresh session):**
1. `cat state/experiment_state.json state/NEXT_ACTION.md`; `pytest tests/ -q` (expect 41 pass).
2. Build `src/j_pretrain/training/` + `src/j_pretrain/artifacts/`:
   - Resumable trainer: AdamW (b1 .9, b2 .95, clip 1.0, bf16), cosine+warmup, grad-accum to hit
     frozen global-batch tokens; dataloader over PackedDataset + lambda interleave schedule;
     exact per-source token counters; wandb logging (entity ametind-o, project j-pretrain).
   - Atomic checkpoint writer: analysis-snapshot + full-resumable classes (tmp->fsync->checksum->
     load-test->rename); metadata schema per CLAUDE.md; RNG (py/np/torch cpu+cuda) + dataloader
     cursor + shuffle state in resumable. Append-only inventory (artifacts/checkpoint_inventory.jsonl).
   - Checkpoint schedules (analysis + resumable) from mission doc.
3. Deterministic resume integration test (train N vs train K->save->resume->N; losses+params match).
4. THEN storage bench (serialize 1 analysis + 1 resumable ckpt, measure sizes) -> STORAGE_PLAN.md;
   finalize COMPUTE_PLAN + reports/FEASIBILITY.md (21-day / storage gate).
5. THEN orchestration (DAG, GPU lock, tmux durable runner, process_registry/run_queue).
6. THEN PRETRAIN_READINESS_AUDIT (subagent) -> SCOPE_LOCK.json -> launch full build_datasets.py
   detached (tokenize 8.7B C4 + 320M MP + 220M ChemPile; ~17GB+; hours) -> then Stage 1.

**Must NOT repeat:** preflight, env, spec, benchmark, config/model, data pipeline+tests, tokenizer
freeze, revision pin, DATA_AUDIT (all done).
**Must NOT:** commit weights/datasets/*.npy/logs; use `conda run`; reduce scope; train before
readiness audit + scope lock + feasibility gate.

**Command to resume orchestrator:** none yet (orchestrator not built).
**Command to build datasets (full, later, detached):** `HF_HUB_ENABLE_HF_TRANSFER=0
TOKENIZERS_PARALLELISM=false J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts
/home/hshi-j-4090/miniconda3/envs/jpre/bin/python scripts/build_datasets.py`
