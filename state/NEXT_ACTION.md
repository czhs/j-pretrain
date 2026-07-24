# NEXT ACTION

**Phase:** stage1_training_lambda0. Datasets BUILT (6 manifests), probes BUILT (256 windows each),
orchestrator RUNNING. Training `music-300m_lambda-0.0` Stage 1 (last, ~1h from stage1 completion).

**Stage-2 config FROZEN 2026-07-24** (preregistered lambda=0 pilot; DECISIONS.md + configs/stage2/music.json
provenance=LOCAL_FROZEN_FIG3A_PREREG_PILOT). Values: LR5e-4/dropout0/batch480/warmup500/WD0.1/minLR5e-5/
max2B/ES-patience3/seed1234. scientific_dict hash 747ff3e5. Documentary edit only — run config_hash UNCHANGED
(provenance & _doc excluded from hash). Applied IDENTICALLY to all 5 lambdas. **No further Stage-2 config
action needed** — orchestrator will consume the frozen file automatically at the flip.

**Last verified 2026-07-24T05:20Z:** stage1 lambda=0 opt_step ~16217, driver 8.502B/8.7B C4 (97.7%),
loss ~3.18, no errors (`logs/orchestrator_errors.jsonl` absent), disk 1052G free, GPU 99%/76C, etime 1d21h46m.
Rate ~0.186B/h → ~0.2B left → stage1 lambda=0 ETA ~2026-07-24T06:20Z (~1h). Tests: 99 pass (CPU-only).

**Process running?** YES — orchestrator, tmux `orch`, pid 1241384 (GPU). Log `logs/orch_20260722T072111Z.log`.
Errors `logs/orchestrator_errors.jsonl`. GPU lock `state/gpu.lock`. **DO NOT restart a healthy run.** ONE GPU job.

**CRITICAL OPS:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY (NOT `conda run`).
Env: `HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`.

**Health-check (~hourly) — one pass, then update state + WAIT_HINT + end session:**
1. `ps -p 1241384 -o pid=,etime=,stat=` + `cat state/gpu.lock`; `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`.
2. `tail -5 logs/orchestrator_errors.jsonl 2>/dev/null`; `jq -c '.runs' state/experiment_state.json`.
3. `tail -1 $ARTIFACT_ROOT/run_metrics/music-300m_lambda-0.0__stage1.jsonl`; `tail -3 artifacts/checkpoint_inventory.jsonl | cut -c1-160`.
4. `df -BG --output=avail $ARTIFACT_ROOT | tail -1` (thresholds 250/150/75G).

**On events (act immediately):**
- stage1 lambda=0 `running`→`complete`, then lambda=0 **stage2 starts** (DAG runs full lambda=0 pipeline
  s1→s2→s3 BEFORE any nonzero-lambda). Normal — keep monitoring. Commit milestone "Complete stage1 music-300m lambda-0.0".
- **When lambda=0 stage2 COMPLETES → run the preregistered PILOT PASS CHECK** (DECISIONS.md 2026-07-24):
  MusicPile val loss finite throughout, decreased below incoming Stage-1→MP eval, reached ES/converged min.
  PASS → record in DECISIONS/ledger, continue (config already frozen). FAIL → NaN/divergence playbook, revise
  config, rerun ALL conditions. This is the last Stage-2 config decision gate.
- Process DEAD or errors present → read error, diagnose; relaunch via resume_cmd in `state/process_registry.json`
  (resume-safe, never restarts healthy/complete node). Retry ≤3/node then failed_blocked.
- ALL nodes complete → per-run audits (auto), then analysis: results/, figures/, reports, AUDIT, `python scripts/verify_completion.py`.

**Wait mechanics:** WAIT_HINT=900 (stage1 ETA ~1h). WAKE_WHEN armed on `logs/orchestrator_errors.jsonl` matching `error`.

**Must NOT:** restart healthy run; run 2 GPU jobs; commit weights/datasets/*.npy/logs/wandb runs; use `conda run`;
reduce scope; edit Stage-2 numeric config (FROZEN). build_datasets already EXITED 0 — do not relaunch.
