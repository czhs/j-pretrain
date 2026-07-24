# NEXT ACTION

**Phase:** stage3_training_lambda0. Stage1 lambda=0 COMPLETE+audited. **Stage2 lambda=0 COMPLETE+audited** (per_run_audits/...__stage2.json ok:true). **Preregistered pilot: PASS** (DECISIONS.md 2026-07-24; L_im=2.2311, finite throughout, dropped 3.5907→2.2311, ES at step3264 restored best step2958). Stage-2 config STANDS unchanged for all 5 lambdas. **Stage3 lambda=0 RUNNING** (ChemPile FT, lr 5e-5, 200M tok budget).

**Last verified 2026-07-24T~16:50Z** (etime 2d09h off start 07-22T07:21:12Z): stage3 opt_step 4, driver 1.97M/200M ChemPile tok, train_loss 3.199, lr warming (2e-6→up to 5e-5). GPU 99%/76C 16.3G. pid 1241384 alive. No errors (`logs/orchestrator_errors.jsonl` empty). Disk 939G free (>250G ok). Stage3 analysis snapshots incoming/tok1M/tok3M already in inventory.

**Process running?** YES — orchestrator, tmux `orch`, pid 1241384 (GPU). Log `logs/orch_20260722T072111Z.log`. Errors `logs/orchestrator_errors.jsonl`. GPU lock `state/gpu.lock`. **DO NOT restart a healthy run.** ONE GPU job.

**CRITICAL OPS:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY (NOT `conda run`).
Env: `HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`.

**Health-check (~hourly) — one pass, then update state + WAIT_HINT + end session:**
1. `ps -p 1241384 -o pid=,etime=,stat=` + `cat state/gpu.lock`; `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`.
2. `tail -5 logs/orchestrator_errors.jsonl 2>/dev/null`; `jq -c '.runs' state/experiment_state.json`.
3. `tail -1 $ARTIFACT_ROOT/run_metrics/music-300m_lambda-0.0__stage3.jsonl`; `tail -3 artifacts/checkpoint_inventory.jsonl | cut -c1-160`.
4. `df -BG --output=avail $ARTIFACT_ROOT | tail -1` (thresholds 250/150/75G).

**On events (act immediately):**
- **stage3 lambda=0 COMPLETE → per-run audit runs auto.** Then evaluate L_ret/L_ft/L_pre at final stage3 ckpt (orchestrator does final eval). DAG then proceeds to **lambda=0.25 stage1** (the big node: 8.7B C4 + 75M MP tok, ~2-3 days). This flip is routine — keep monitoring; no config decisions remain (Stage-1/2/3 all FROZEN).
- Process DEAD or errors present → read error, diagnose; relaunch via resume_cmd in `state/process_registry.json` (resume-safe, never restarts healthy/complete node). Retry ≤3/node then failed_blocked.
- ALL nodes complete → per-run audits (auto), then analysis: results/, figures/, reports, final AUDIT (fresh subagent), `python scripts/verify_completion.py`.

**Wait mechanics:** stage3 ETA ~1-2h (200M tok). WAKE_WHEN armed on stage3 metrics for stage3→lambda0.25-stage1 flip is NOT needed (routine); arm on stage1 lambda-0.25 metrics file instead OR just WAIT_HINT=3600 for hourly checks.

**Must NOT:** restart healthy run; run 2 GPU jobs; commit weights/datasets/*.npy/logs/wandb runs; use `conda run`; reduce scope; edit any Stage config (ALL FROZEN). Stage-2 pilot already PASSED — no further config gates.
