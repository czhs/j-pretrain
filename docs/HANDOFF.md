# HANDOFF — exact operating commands

Everything a fresh operator (human or Claude session) needs to install, inspect, resume, verify,
and regenerate this experiment. Every command below is copy-pasteable as written.

**Scope reminder:** this is a deliberately reduced Figure 3a — 1 MusicPile subset size (300M) ×
lambda ∈ {0, 0.25, 0.5, 0.75, 1.0}. The paper sweeps 30M/150M/300M; only 300M is reproduced here.
Authorized in advance by the user; recorded in `state/SCOPE_LOCK.json`. NOT Figure 3b.

## 0. Shell prelude (required for every GPU / HF / artifact command)

```bash
cd /home/hshi-j-4090/Desktop/j-pretrain
export PY=/home/hshi-j-4090/miniconda3/envs/jpre/bin/python
export J_PRETRAIN_ARTIFACT_ROOT=/home/hshi-j-4090/Desktop/j-pretrain-artifacts
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
```

Invoke `$PY` **directly**. Do not use `conda run` (it buffers/steals stdio and has caused detached
runs to be misclassified). Never install experiment deps into conda `base`.

## 1. Install the environment

```bash
bash scripts/setup_env.sh          # creates/updates conda env `jpre` with the pinned stack
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Pins live in `scripts/setup_env.sh` and `pyproject.toml`; the recorded host/driver/CUDA/kernel
facts are in `docs/ENVIRONMENT.md`. Do not upgrade a pinned dependency during an active run.

## 2. Verify datasets and manifests

Tokenized shards and manifests are under `$J_PRETRAIN_ARTIFACT_ROOT/datasets/<source>/<split>/`
(sources: `c4`, `musicpile`, `chempile`; splits: `train`, `val`). They are **not** in Git.

```bash
# all six (source, split) manifests, with revision + token counts + frozen tokenizer hash
find "$J_PRETRAIN_ARTIFACT_ROOT/datasets" -name manifest.json \
  -exec sh -c 'echo "== $1"; jq -c "{source,split,revision,n_tokens,n_seqs,seq_len,tokenizer_sha256}" "$1"' _ {} \;

# frozen probe manifest (fixed probes, identical token IDs for every checkpoint)
jq -c 'to_entries[] | {source:.key, n_windows:.value.n_windows, seq_len:.value.seq_len,
       revision:.value.revision, tokenizer_sha256:.value.tokenizer_sha256}' \
  "$J_PRETRAIN_ARTIFACT_ROOT/probes/probe_manifest.json"

# manifest hashes recorded in canonical state must match the built manifests
jq '.dataset_manifest_hashes' state/experiment_state.json
```

Rebuild from scratch only if manifests are missing or a hash mismatches (this is expensive and
must reproduce byte-identical manifests — the pipeline is deterministic given the recorded seeds):

```bash
$PY scripts/build_datasets.py            # add --smoke for a tiny path-validation run
$PY scripts/build_probes.py
```

Data-integrity properties (split disjointness, no ChemPile in Stage 1/2, no val leakage, exact
lambda token allocation, 300M subset size, tokenizer determinism) are enforced by tests, not by
eyeballing manifests — see §8. Findings are written up in `docs/DATA_AUDIT.md`.

## 3. Inspect experiment state

```bash
cat state/NEXT_ACTION.md                      # what to do right now (<50 lines)
jq '{phase,runs,next_action,artifact_root}' state/experiment_state.json
tail -n 1 state/iteration_ledger.jsonl | jq .
jq '.processes[] | {name,classification,pid,tmux_session,current_node,log}' state/process_registry.json
jq . state/SCOPE_LOCK.json
cat state/gpu.lock 2>/dev/null || echo "no GPU lock held"
```

Live training progress (bounded output only — never cat a whole metrics stream):

```bash
tail -n 2 "$J_PRETRAIN_ARTIFACT_ROOT/run_metrics/music-300m_lambda-0.0__stage1.jsonl"
tail -n 5 logs/orchestrator_errors.jsonl 2>/dev/null || echo "no node errors"
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
df -BG --output=avail "$J_PRETRAIN_ARTIFACT_ROOT" | tail -1   # thresholds: 250 / 150 / 75 GB free
```

## 4. Resume the orchestrator

The orchestrator walks the run DAG, takes an exclusive GPU lock, skips already-validated
artifacts, and resumes partial runs from the latest valid resumable checkpoint. It is
**resume-safe**: relaunching never restarts a healthy or completed node.

First confirm nothing is already running (ONE GPU job at a time):

```bash
tmux ls 2>/dev/null; cat state/gpu.lock 2>/dev/null
```

If no live orchestrator holds the lock:

```bash
tmux new-session -d -s orch "HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false \
  J_PRETRAIN_ARTIFACT_ROOT=$J_PRETRAIN_ARTIFACT_ROOT \
  $PY -m j_pretrain.orchestration.run 2>&1 | tee logs/orch_\$(date -u +%Y%m%dT%H%M%SZ).log"
tmux capture-pane -p -t orch | tail -20
```

The canonical copy of this command is `state/process_registry.json → processes[].resume_cmd`.

Options: `--max-nodes N` (run at most N DAG nodes then exit), `--no-wandb`, `--artifact-root PATH`.

A stale lock (PID in `state/gpu.lock` no longer exists) is detected and cleared automatically on
the next launch. Do **not** kill a running orchestrator because a Claude session ended.

## 5. Resume an individual run

There is no per-run entrypoint by design — resumption is a property of the DAG, not of a script.
To advance exactly one node (e.g. after diagnosing a single failure):

```bash
$PY -m j_pretrain.orchestration.run --max-nodes 1
```

The orchestrator selects the first ready node, validates its config hash and parent-checkpoint
lineage against the frozen configs, loads the latest valid resumable checkpoint (model + AdamW +
scheduler + RNG states + data-loader cursor), and continues. Transient failures retry ≤3 times per
node; after that the node is marked `failed_blocked` and a failure report is written. Never
hand-edit a run's status to force a restart past a valid resume point.

## 6. Verify checkpoints

```bash
# inventory: append-only, one JSON record per checkpoint / prune event
wc -l artifacts/checkpoint_inventory.jsonl
tail -n 3 artifacts/checkpoint_inventory.jsonl | jq -c '{op,run_id,checkpoint_id,stage,checkpoint_class,milestone_labels,total_tokens,load_validation_status,backup_status}'

# checksum + load test for one checkpoint directory
$PY - <<'EOF'
import json, os
from pathlib import Path
from j_pretrain.artifacts.checkpoint import verify_checksums, load_weights
d = Path(os.environ["J_PRETRAIN_ARTIFACT_ROOT"])
recs = [json.loads(l) for l in Path("artifacts/checkpoint_inventory.jsonl").read_text().splitlines() if l.strip()]
rec = [r for r in recs if r.get("op") != "prune"][-1]
p = d / rec["rel_path"]
print(rec["checkpoint_id"], "checksums_ok=", verify_checksums(p), "n_tensors=", len(load_weights(p)))
EOF
```

Full checkpoint auditing (every required milestone present, every checksum verifying, final/best
checkpoints load, lineage consistent, pruning recorded as superseding records) is performed by the
completion verifier in §9 — that is the authoritative check.

**Retention invariant:** analysis snapshots and all non-checkpoint research artifacts are
permanent. Never delete, overwrite, quantize, or dedup them. Only *superseded* intermediate
resumables may be pruned, always keeping ≥2 recent per active run, and every pruning must append a
superseding inventory record.

## 7. Regenerate results and figures

Both come from one deterministic, wandb-free script that reads only the committed inventory and
canonical state. No manual data entry is involved anywhere.

```bash
$PY scripts/analyze_results.py
```

Writes `results/results.csv`, `results/results.json`,
`figures/figure3a_replication.png`, `figures/figure3a_replication.pdf`.

Conditions that have not yet produced metrics appear as empty cells — values are never fabricated.

## 8. Run tests

Run CPU-only so a live GPU training job is undisturbed:

```bash
CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/ -q
```

This covers lambda token allocation, per-source token counting, the fixed 300M subset, split
separation and leakage, dataset/tokenizer determinism, parameter count, optimizer and scheduler,
token-weighted evaluation loss, atomic checkpoint writes and loading, resume/RNG/data-cursor
restoration, the deterministic resume integration test, DAG and run-lock behavior, inventory
append semantics, and the completion verifier's own behavior.

## 9. Run the completion verifier

```bash
$PY scripts/verify_completion.py; echo "exit=$?"
```

Exits nonzero on any unmet mandatory condition. Writes `reports/completion_verification.json` plus
a human-readable summary to stdout. Inspect failures with:

```bash
jq -r '.checks[] | select(.ok==false) | "\(.category)\t\(.name)\t\(.detail)"' reports/completion_verification.json
```

The completion promise may be emitted only when this exits 0, all audits pass with no unresolved
critical/major findings, and the final commit is pushed to `origin/main`.

## 10. Git

```bash
git status --short
git log --oneline -5
git push origin main
```

Commit code, configs, state, docs, small results, figures, inventories. **Never** commit weights,
datasets, tokenized shards, `*.npy`, multi-GB logs, wandb run directories, or secrets.
Never `git push --force*`, `git reset --hard`, `git clean -fd`, or rewrite history.

## 11. If something is broken

1. Read `docs/FAILURE_PLAYBOOK.md` (OOM, NaN/divergence, data-loader failure, power loss).
2. Record the incident in `docs/FAILURES.md`.
3. External blocker → write `BLOCKED.md`, commit, push. Usage-limit pauses and long runtimes are
   **not** blockers and must never trigger `BLOCKED.md` or a scope reduction.
