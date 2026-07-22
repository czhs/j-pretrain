# NEXT ACTION

**Phase:** stage1_training_lambda0. Datasets BUILT (6 manifests, EXIT_CODE=0), probes BUILT
(256 windows each), orchestrator LAUNCHED. Now training `music-300m_lambda-0.0` Stage 1.

**Process running?** YES — orchestrator, tmux `orch`, pid 1241384 (GPU). Log:
`logs/orch_20260722T072111Z.log`. Errors: `logs/orchestrator_errors.jsonl`. GPU lock `state/gpu.lock`.
Confirmed healthy at launch: GPU 99% util, 6455 MiB, 77°C. Shared init ckpt created
(`init-an-seed1234-7f41081e` analysis, `init-re-seed1234-178e52c9` resumable, both in inventory).
**DO NOT restart a healthy run.** ONE GPU job at a time.

**CRITICAL OPS:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY (NOT `conda run`).
Env for GPU/HF ops: `HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`.

**Health-check (~hourly) — one pass, then update state + WAIT_HINT + end session:**
1. `ps -p 1241384 -o pid=,etime=,stat=` (alive?) and `cat state/gpu.lock`.
2. `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader` (util>0 = training).
3. `tail -5 logs/orchestrator_errors.jsonl 2>/dev/null` (any node error?).
4. Node status: `jq '.runs' state/experiment_state.json`. Metrics:
   `ls $ARTIFACT_ROOT/run_metrics/`; ckpts: `tail -3 artifacts/checkpoint_inventory.jsonl | cut -c1-160`.
5. Disk: `df -BG --output=avail $ARTIFACT_ROOT | tail -1` (thresholds 250/150/75G free).

**On events (act immediately):**
- Node status flips `stage1`→`complete` and stage2 begins → normal, continue monitoring.
- Process DEAD or `orchestrator_errors.jsonl` has entries → read error, diagnose; relaunch via
  resume_cmd in `state/process_registry.json` (resume-safe: detects latest valid resumable, never
  restarts healthy/complete node). Retry policy: ≤3 per node then failed_blocked.
- ALL nodes complete (orchestrator prints final JSON, process exits, lock released) → run
  per-run audits (already auto-run), then analysis: `results/`, `figures/`, reports, final AUDIT,
  `python scripts/verify_completion.py`.

**Wait mechanics:** WAIT_HINT=3600 (Stage 1 ETA ~46h/run; hourly checks). WAKE_WHEN armed on
`logs/orchestrator_errors.jsonl` matching `error` (early wake on failure). Update/clear both when phase changes.

**Orchestrator facts:** run_ids `music-300m_lambda-{0.0,0.25,0.5,0.75,1.0}`, seed 1234, ADD policy.
Stage1 lambda=0 ≈16593 steps (8.7B C4 tok, eager mb8). Stage2 FULL 292968-window MP subset every
lambda. Stage3 200M ChemPile (pool 118M → ~1.69 epochs, ShuffledSourcePlan cycles; see DECISIONS).

**Must NOT:** restart healthy run; run 2 GPU jobs; commit weights/datasets/*.npy/logs/wandb runs;
use `conda run`; reduce scope. build_datasets (pid 1100073) already EXITED 0 — do not relaunch it.
