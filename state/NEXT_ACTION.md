# NEXT ACTION

**Phase:** stage1_training_lambda0.25. **⏸ RE-SHELVED 2026-07-24 (manual, free GPU for another
user) — NOT a crash. Orchestrator pid 1468831 and the Ralph DRIVER `loop.sh` (detached shell,
NOT in tmux) are STOPPED; `.claude/ralph-loop.local.md` active:false; `state/gpu.lock` cleared.**
NOTE: an earlier incomplete shelve (tmux-only) let `loop.sh` auto-relaunch training; that iteration
also found+fixed a DAG-reschedule bug (see staged run.py/tests, uncommitted). Last resumable ckpt
`stage1-rs-scheduled-tok500170752-step954`; ran to ~step1144 then stopped (~0 compute lost).
**To resume:** `bash restart_run.sh` (relaunches driver in tmux `ralph`; resumes orchestrator byte-exact).

**FULL lambda=0 condition COMPLETE + audited** (stage1+2+3): L_im=2.2311, **L_ret=2.3948**,
L_ft=2.1574, L_pre=3.7238, forgetting=0.1637.

**Bug found + fixed this iteration (orchestrator DAG):** a node left in a non-terminal status
(`running`/`failed_retryable`/…) was NEVER rescheduled — `dag.is_ready` only schedules
`planned` — so the shelved lambda-0.25 stage1 was orphaned and the first relaunch silently
skipped ahead to lambda-0.5 stage1 (killed after ~3 min; wrote NO artifacts/metrics/inventory
records — clean). Fix in `orchestration/run.py`: `reclaim_stale_nodes()` runs at startup while
the GPU lock is held (⇒ no live owner ⇒ stale by definition) and returns such nodes to
`planned`; retry counts now persist in `experiment_state.json["retry_counts"]` (limit 3, then
`failed_blocked`). Reclaims logged to `logs/orchestrator_reclaims.jsonl`. 2 regression tests
added; full suite 101 passed.

**Process running?** YES — orchestrator, tmux `orch`, **pid 1468831** (GPU).
Log `logs/orch_20260724T221742Z.log`. GPU lock `state/gpu.lock`. **DO NOT restart a healthy
run.** ONE GPU job at a time.

**Last verified 2026-07-24T22:22Z:** GPU 99% / 17.9 GiB. lambda-0.25 stage1 at
driver_tokens ~511M/8775M (~5.8%), opt_step 975, train_loss ~4.78, grad_norm ~0.55.
Disk 920G avail (>250G ok). WAIT_HINT=3600. WAKE_WHEN armed on `"event": "final"` in the
lambda-0.25 stage1 metrics file (completion event, NOT mere file existence).

**CRITICAL OPS:** Use `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python` DIRECTLY (NOT `conda run`).
Env: `HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts`.

**Health-check (~hourly) — one pass, then update state + WAIT_HINT + end session:**
1. `ps -p 1468831 -o pid=,etime=,stat=`; `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`.
2. `tail -5 logs/orchestrator_errors.jsonl`; `jq -c '.runs' state/experiment_state.json`.
3. `tail -1 $ART/run_metrics/music-300m_lambda-0.25__stage1.jsonl`; `tail -3 artifacts/checkpoint_inventory.jsonl | cut -c1-160`.
4. `df -BG --output=avail $ART | tail -1` (thresholds 250/150/75G).

**On events (act immediately):**
- lambda-0.25 stage1 COMPLETE → per-run audit auto → DAG → lambda-0.25 stage2 (MusicPile, ES) → stage3 → then 0.5, 0.75, 1.0 (each stage1→2→3).
- Process DEAD → relaunch `bash restart_run.sh` (resume-safe; now also reclaims stale nodes).
- ALL nodes complete → analysis: results/, figures/, reports, final AUDIT (fresh subagent), `python scripts/verify_completion.py`.

**Must NOT:** restart healthy run; run 2 GPU jobs; commit weights/datasets/*.npy/logs/wandb runs;
use `conda run`; reduce scope; edit any Stage config (ALL FROZEN).
