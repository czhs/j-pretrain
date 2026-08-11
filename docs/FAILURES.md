# FAILURES.md — running log of failures & non-obvious issues

Append-only chronological record. Each entry: date, symptom, root cause, resolution, whether any
run/artifact was invalidated. Routine status lives in the ledger, not here.

## 2026-07-21 — `conda run -n jpre` swallows heredoc/stdout
- **Symptom:** Python invoked via `conda run -n jpre python - <<'PY'` produced no stdout for data
  probes, making it look like data access failed.
- **Root cause:** `conda run` buffers/redirects child stdout in a way that drops heredoc output here.
- **Resolution:** Always invoke the interpreter directly:
  `/home/hshi-j-4090/miniconda3/envs/jpre/bin/python`. Recorded in NEXT_ACTION as a critical ops note.
- **Impact:** none (tooling only; no run/artifact affected).

## 2026-07-21 — `hf_hub_download` hung once during preflight
- **Symptom:** a single `hf_hub_download` call hung; `curl` to the same host worked.
- **Root cause:** not fully diagnosed; likely `hf_transfer`/connection stall.
- **Resolution:** set `HF_HUB_ENABLE_HF_TRANSFER=0` for all HF/data ops; streaming + metadata access
  via `datasets`/`HfApi` works reliably. Dataset revisions were pinned successfully afterward.
- **Impact:** none (transient; datasets reachable, revisions pinned).

## 2026-07-21 — safetensors stores tied embedding twice (+56 MB/analysis snapshot)
- **Symptom:** analysis snapshot measured 310.6 MB vs naive 269 MB expectation.
- **Root cause:** `safetensors` materialises both `embed_tokens.weight` and `lm_head.weight` for the
  tied 135M model (they share storage in-memory but are written as two tensors).
- **Resolution:** accepted — load-independence and "complete unquantized weights" outweigh the 56 MB;
  folded into the storage projection (docs/STORAGE_PLAN.md). No action needed.
- **Impact:** storage projection uses the true 310.6 MB figure; gate still PASS.

<!-- New failures appended below this line. -->

## 2026-07-24 — Orphaned DAG node after manual shelve (orchestrator scheduling bug)

**Symptom.** After un-shelving (GPU freed, `restart_run.sh`), the orchestrator started
`music-300m_lambda-0.5::stage1` instead of resuming the interrupted
`music-300m_lambda-0.25::stage1` (which was ~4.4% done).

**Root cause.** `dag.is_ready()` schedules a node only when its status is exactly `planned`.
The manual shelve killed the process while `experiment_state.json` still recorded
`lambda-0.25::stage1 = "running"`, so that node could never become ready again — the DAG
skipped it permanently and advanced to the next `planned` node. The same defect made the
`failed_retryable` retry path dead: a node set to `failed_retryable` was never re-scheduled,
so the documented "retry ≤3 times" behaviour never actually retried.

**Impact.** None scientific. The lambda-0.5 job was killed after ~3 minutes, before any
checkpoint, metric line, or inventory record was written (verified: no
`checkpoints/music-300m_lambda-0.5/`, no `run_metrics/...lambda-0.5__stage1.jsonl`, 0
inventory rows). No artifact was created, overwritten, or deleted.

**Fix.** `orchestration/run.py`:
* `reclaim_stale_nodes()` runs once at orchestrator startup, *after* the GPU lock is
  acquired — holding the lock proves no live trainer owns any node, so every non-terminal
  status (`ready/running/checkpointing/evaluating/complete_unverified/failed_retryable`) is
  stale by construction and is returned to `planned`.
* Retry counts moved from an in-process dict to `experiment_state.json["retry_counts"]`, so
  the ≤3 limit survives restarts; past the limit a node becomes `failed_blocked` instead of
  being reclaimed in a loop.
* Retry paths now set `planned` (not `failed_retryable`) so the node is actually retried.
* Every reclaim is appended to `logs/orchestrator_reclaims.jsonl`.

Rescheduling never re-trains finished work: `run_node()` resume-detects the latest valid
resumable and continues byte-exact, and a stage already at `total_steps` short-circuits via
the crash-after-final guard.

**Verification.** Two regression tests in `tests/test_orchestrator_run.py`
(`test_stale_running_node_is_reclaimed_and_resumed`,
`test_reclaim_blocks_node_past_retry_limit`); full suite 101 passed. In production the
relaunch logged both reclaims and resumed lambda-0.25 stage1 at opt_step 954 → 975.
