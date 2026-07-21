# FAILURE_PLAYBOOK.md

Procedures for recovering from failures WITHOUT corrupting scientific validity or losing artifacts.
Golden rules: never restart a healthy detached run; never delete a research artifact to recover;
apply any correction identically to every comparable condition; record every change in
`state/DECISIONS.md` + `docs/FAILURES.md`.

## CUDA OOM
1. Confirm no unrelated GPU process holds VRAM: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
2. Record current peak allocated/reserved from the run's metrics.
3. Lower `ExecConfig.microbatch_size` (execution-only; scientific hash unchanged).
4. Increase grad-accum to preserve the frozen effective global batch (`StageConfig.global_batch_seqs`;
   the driver derives `grad_accum = global_batch_seqs / microbatch_size` — keep it an exact divisor).
5. Enable `gradient_checkpointing=True` if still tight.
6. Resume from the latest valid resumable checkpoint (never restart from step 0).
7. Record the execution change (this NEVER changes any scientific hyperparameter).
Benchmark headroom: compile@mb8 uses 9.65 GB reserved of 24 GB, so OOM is unlikely; fallback is
eager SDPA @ mb8 (16.6 GB) or mb4 (9.5 GB).

## NaN / divergence
1. Stop the affected run (do not let it overwrite good checkpoints).
2. Preserve the failure checkpoint + logs (they are evidence; never delete).
3. Reproduce on a short deterministic segment from the last good resumable.
4. Check in order: corrupt token ids (>= vocab), padding/label masks, loss scaling, LR schedule,
   optimizer-state restoration, grad clipping, BF16 overflow (bf16 has fp32 range → rare).
5. Fix root cause; apply the fix to EVERY comparable condition.
6. Decide + document whether earlier runs are invalidated (never silently drop a condition).

## Data-loader / shard failure
Data order is a pure function of the integer cursor (`dataplan`), so recovery is deterministic:
record shard name, window/token offset, worker, exception, and `windows_consumed`. Repair the shard
(re-tokenize that (source,split) — `build_datasets.py` is resumable via manifest skip), verify its
sha256 against the manifest, and resume; the same logical order is reproduced from the cursor.

## Power loss / process death
On restart: read canonical state → inspect `state/gpu.lock` (reclaim if owner pid dead) and
`process_registry.json` → validate the latest COMPLETE resumable (ignore `*.tmp` writes) →
resume from it → preserve all failure evidence. `is_complete()` requires the `.complete` marker, so
half-written checkpoints are never mistaken for valid.

## Disk pressure (absolute free-space thresholds)
- **250 GB free**: recompute the full storage projection (`scripts/bench_storage.py`) and confirm
  remaining runs + artifacts fit.
- **150 GB free**: do not launch a new run unless its full permanent-artifact projection + margin fits.
- **75 GB free**: finish the in-flight atomic checkpoint write → pause orchestrator → save a resumable
  if space allows → update state → write `BLOCKED.md` requesting more storage.
Never delete analysis snapshots, permanently-retained resumables, logs, metrics, manifests, probes,
failure records, or checksums to relieve pressure. Only regenerable package/download caches OUTSIDE
the artifact tree may be removed, and each removal is recorded.

## Transient vs deterministic failures
Orchestrator retries a transient failure at most 3× (`retry_counts`), then marks the node
`failed_blocked`, writes a failure report, and moves to the next ready DAG node. A deterministic
failure (same error every attempt) is never retried blindly.

## Usage-limit pauses
NOT a blocker. Training is detached (tmux) and survives arbitrary orchestration gaps. On resume,
follow the standard beginning-of-iteration protocol; never restart a run, kill a process, or repeat
completed work because of a gap. Do not create `BLOCKED.md` for quota.

## wandb failures
Supplementary only. On any wandb error, fall back to `WANDB_MODE=offline` (already handled in
`orchestration/metrics.py`) and continue; local JSONL metrics remain the canonical record. Never a blocker.
