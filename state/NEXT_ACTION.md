# NEXT ACTION

> **⏸ SHELVED 2026-07-24 (manual, to free the GPU for another user).** Not a crash.
> Ralph loop (`early-exposure`) and orchestrator (`orch`, pid 1241384) tmux sessions
> were killed; `state/gpu.lock` cleared. Last resumable ckpt: `stage1-rs-scheduled-tok500170752-step954`
> (~0 compute lost). **To resume:** `bash restart_run.sh` (byte-exact resume), then
> optionally restart the Ralph loop. Everything below reflects pre-shelve state.


**Phase:** stage1_training_lambda0.25. **FULL lambda=0 condition COMPLETE + audited** (stage1+2+3). Stage3 lambda=0 final metrics (per_run_audits/...__stage3.json ok:true, findings []):
- **L_ret (val_mp) = 2.3948**, L_ft (val_chempile) = 2.1574, L_pre (val_c4) = 3.7238
- L_im (stage2) = 2.2311, **forgetting = L_ret - L_im = 0.1637**
- final ckpts verified: `stage3-an-final-...-361e79b2` (analysis), `stage3-rs-final-...-09c4396b` (resumable).

**DAG advanced → lambda=0.25 stage1 TRAINING** (C4 8.7B + 75M MP interleaved). This is the LONG node (~1-2 days). No config gates remain — all Stage-1/2/3 configs FROZEN.

**Last verified 2026-07-24T21:12Z:** pid 1241384 alive (Rl+, etime 2d13h51m), GPU 99% 16.8G 76C. lambda=0.25 stage1 at driver_tokens ~390M/8775M (~4.4%), opt_step 745, train_loss ~5.09, grad_norm ~0.82. Errors file empty. Disk 922G avail (>250G ok). WAIT_HINT=3600 (next actionable event is ~a day+ away). WAKE_WHEN armed on `"event": "final"` in lambda-0.25 stage1 metrics (completion event, NOT mere file existence).

**Process running?** YES — orchestrator, tmux `orch`, pid 1241384 (GPU). Log `logs/orch_20260722T072111Z.log`. Errors `logs/orchestrator_errors.jsonl`. GPU lock `state/gpu.lock`. **DO NOT restart a healthy run.** ONE GPU job.

**CRITICAL OPS:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY (NOT `conda run`).
Env: `HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`.

**Health-check (~hourly) — one pass, then update state + WAIT_HINT + end session:**
1. `ps -p 1241384 -o pid=,etime=,stat=` + `cat state/gpu.lock`; `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`.
2. `tail -5 logs/orchestrator_errors.jsonl`; `jq -c '.runs' state/experiment_state.json`.
3. `tail -1 $ARTIFACT_ROOT/run_metrics/music-300m_lambda-0.25__stage1.jsonl`; `tail -3 artifacts/checkpoint_inventory.jsonl | cut -c1-160`.
4. `df -BG --output=avail $ARTIFACT_ROOT | tail -1` (thresholds 250/150/75G).

**On events (act immediately):**
- lambda=0.25 stage1 COMPLETE → per-run audit auto → DAG proceeds to lambda=0.25 stage2 (MusicPile post-train, ES). Then stage3. Then lambda=0.5, 0.75, 1.0 (each stage1→2→3).
- Process DEAD or errors present → read error, diagnose; relaunch via resume_cmd in `state/process_registry.json` (resume-safe, never restarts healthy/complete node). Retry ≤3/node then failed_blocked.
- ALL nodes complete → per-run audits (auto), then analysis: results/, figures/, reports, final AUDIT (fresh subagent), `python scripts/verify_completion.py`.

**Must NOT:** restart healthy run; run 2 GPU jobs; commit weights/datasets/*.npy/logs/wandb runs; use `conda run`; reduce scope; edit any Stage config (ALL FROZEN).
