# NEXT ACTION

**Phase:** stage2_training_lambda0. Stage1 lambda=0 **COMPLETE + audited** (per_run_audits/music-300m_lambda-0.0__stage1.json ok:true). Stage2 lambda=0 **RUNNING** on MusicPile (full 300M subset, max 2B tok, ES-patience3).

**Last verified 2026-07-24T07:00Z:** stage1 lambda=0 final val_mp=3.5907, val_c4=3.1975, val_chempile=3.3422, peak_vram 15.25G. Stage2 opt_step 2, driver 0.98M/2B MP tok, loss 3.56 (incoming baseline 3.59, decreasing — healthy). Incoming stage2 analysis+resumable ckpts saved. No errors (`logs/orchestrator_errors.jsonl` empty). Disk 1048G free. GPU 96%/71C. pid 1241384 etime ~1d23h.

**Process running?** YES — orchestrator, tmux `orch`, pid 1241384 (GPU). Log `logs/orch_20260722T072111Z.log`. Errors `logs/orchestrator_errors.jsonl`. GPU lock `state/gpu.lock`. **DO NOT restart a healthy run.** ONE GPU job.

**CRITICAL OPS:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY (NOT `conda run`).
Env: `HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`.

**Health-check (~hourly) — one pass, then update state + WAIT_HINT + end session:**
1. `ps -p 1241384 -o pid=,etime=,stat=` + `cat state/gpu.lock`; `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`.
2. `tail -5 logs/orchestrator_errors.jsonl 2>/dev/null`; `jq -c '.runs' state/experiment_state.json`.
3. `tail -1 $ARTIFACT_ROOT/run_metrics/music-300m_lambda-0.0__stage2.jsonl`; `tail -3 artifacts/checkpoint_inventory.jsonl | cut -c1-160`.
4. `df -BG --output=avail $ARTIFACT_ROOT | tail -1` (thresholds 250/150/75G).

**On events (act immediately):**
- **stage2 lambda=0 COMPLETE → run preregistered PILOT PASS CHECK** (DECISIONS.md 2026-07-24):
  MP val loss finite throughout, decreased below incoming (3.5907), reached ES/converged min.
  PASS → record in DECISIONS/ledger, continue (config already frozen; DAG proceeds to lambda=0 stage3, then nonzero-lambda s1→s2→s3). FAIL → NaN/divergence playbook, revise config, rerun ALL conditions. Last Stage-2 config gate.
- stage2→stage3 flip (lambda=0): normal. Stage3 ChemPile FT lr 5e-5, 200M tok. Keep monitoring.
- Process DEAD or errors present → read error, diagnose; relaunch via resume_cmd in `state/process_registry.json` (resume-safe, never restarts healthy/complete node). Retry ≤3/node then failed_blocked.
- ALL nodes complete → per-run audits (auto), then analysis: results/, figures/, reports, AUDIT, `python scripts/verify_completion.py`.

**Wait mechanics:** WAIT_HINT chosen from ETA of next actionable event (stage2 completion, unknown but hours → 3600). WAKE_WHEN armed on stage3 metrics file (fires on stage2→stage3 flip).

**Must NOT:** restart healthy run; run 2 GPU jobs; commit weights/datasets/*.npy/logs/wandb runs; use `conda run`; reduce scope; edit Stage-2 numeric config (FROZEN). build_datasets already EXITED 0 — do not relaunch.
