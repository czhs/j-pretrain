# NEXT ACTION

**Phase:** readiness_audited_ready_awaiting_datasets. Orchestrator + probes built; 87 tests pass.
Readiness audit = READY (18 PASS/0 crit, reports/PRETRAIN_READINESS_AUDIT.md). Its one MAJOR
(restored_best used terminal not best-val weights) is FIXED + regression-tested (commit 1aeb68a).
**Only remaining Stage-1 gate: dataset build must finish.**

**Processes running?**
- `build_datasets` tmux `dbuild` pid 1100073 (CPU tokenize, NOT GPU). Log:
  `logs/build_datasets_20260721T133614Z.log`. ~7/85 C4 train shards done (~11min/shard); C4 ~15h
  then MP + ChemPile. Resumable (skips complete manifests). DO NOT restart if healthy. Health-check
  ~10min: `grep -E '\[done\]|EXIT_CODE' <log> | tail; ps -p 1100073 -o etime=`.

**CRITICAL OPS:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY (NOT `conda run`).
For HF/data/GPU ops set `HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`.

**Next exact actions (fresh session):**
1. If readiness-audit report exists: read `reports/PRETRAIN_READINESS_AUDIT.md`; resolve criticals/majors.
2. Poll build. When `grep EXIT_CODE <log>` shows `EXIT_CODE=0`:
   - Confirm 6 manifests: `find $ARTIFACT_ROOT/datasets -name manifest.json` (c4/musicpile/chempile x train/val).
   - Record real dataset_manifest_hashes into state/experiment_state.json (per-manifest sha or n_seqs).
   - Verify MusicPile train pool >= 292968 windows (300M subset) and C4 train >= 8496093 windows.
   - Build probes: `python -m` a small driver using j_pretrain.data.probes.build_probe over each
     val set (256 windows) -> `$ARTIFACT_ROOT/probes/{c4,musicpile,chempile}/` + probe_manifest.json.
3. Launch orchestrator DETACHED in tmux (ONE GPU job): 
   `tmux new-session -d -s orch 'HF_HUB_ENABLE_HF_TRANSFER=0 J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts /home/hshi-j-4090/miniconda3/envs/jpre/bin/python -m j_pretrain.orchestration.run 2>&1 | tee logs/orch_$(date -u +%Y%m%dT%H%M%SZ).log'`
   It auto-creates the shared init ckpt, then runs lambda=0 stage1 first (run_queue order).
4. Monitor: metrics at `$ARTIFACT_ROOT/run_metrics/<run>__<stage>.jsonl`; ckpt inventory at
   `artifacts/checkpoint_inventory.jsonl`; experiment_state runs[*][stage]. GPU lock: `state/gpu.lock`.
   Health-check ~10min; wake on process death / stage completion / disk pressure.

**Orchestrator facts:** run_ids `music-300m_lambda-{0.0,0.25,0.5,0.75,1.0}`, seed 1234, ADD policy.
Stage1 total_steps=(8496093+round(lambda*292968))//512. Stage2/3 total_steps=max_optimizer_steps.
Stage2 FULL 292968-window subset every lambda (Fig 3a). Exec: microbatch 8, torch_compile OFF (eager,
stable+deterministic; feasibility PASS at eager <21d), fused AdamW on cuda, bf16 autocast.

**Must NOT repeat:** preflight, env, spec, benchmark, configs, data pipeline+tests, tokenizer freeze,
revision pin, DATA_AUDIT, training+artifacts+orchestration modules+tests, resume tests, verifier, docs.
**Must NOT:** commit weights/datasets/*.npy/logs/wandb runs; use `conda run`; reduce scope; launch
Stage 1 before readiness audit reviewed + build EXIT_CODE=0 + probes built; restart healthy build.

**Resume orchestrator (after any crash/gap):** same tmux launch as step 3 — it resume-detects the
latest valid resumable per (run,stage) and never restarts a healthy/complete node.
