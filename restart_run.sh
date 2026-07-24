#!/usr/bin/env bash
# restart_run.sh — un-shelve the j-pretrain experiment.
# Relaunches the orchestrator, which resume-detects the latest valid resumable
# checkpoint and continues byte-exact (fp32 model+optimizer+RNG). Safe to run
# multiple times: it refuses to start a second GPU job.
set -euo pipefail

REPO="/home/hshi-j-4090/Desktop/j-pretrain"
ART="/home/hshi-j-4090/Desktop/j-pretrain-artifacts"
PY="/home/hshi-j-4090/miniconda3/envs/jpre/bin/python"
cd "$REPO"

# --- guard: don't double-launch -------------------------------------------------
if tmux has-session -t orch 2>/dev/null; then
  echo "ERROR: tmux session 'orch' already exists. Attach with:  tmux attach -t orch"
  echo "Refusing to launch a second GPU job."
  exit 1
fi
if [ -f state/gpu.lock ]; then
  LOCKPID="$(grep -oE '"pid":[[:space:]]*[0-9]+' state/gpu.lock | grep -oE '[0-9]+' || true)"
  if [ -n "${LOCKPID:-}" ] && kill -0 "$LOCKPID" 2>/dev/null; then
    echo "ERROR: state/gpu.lock is held by live pid $LOCKPID. Another job is running."
    exit 1
  fi
  echo "Note: clearing stale state/gpu.lock (owner not alive)."
  rm -f state/gpu.lock
fi

# --- sanity: GPU largely free ---------------------------------------------------
USEDMB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "${USEDMB:-0}" -gt 4000 ]; then
  echo "WARNING: GPU already using ${USEDMB} MiB. Someone else may still be on it."
  read -r -p "Launch anyway? [y/N] " ans
  [ "${ans:-N}" = "y" ] || { echo "Aborted."; exit 1; }
fi

# --- launch (resume-safe; same command from state/process_registry.json) --------
echo "Launching orchestrator in tmux session 'orch' (byte-exact resume)..."
tmux new-session -d -s orch \
  "HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false J_PRETRAIN_ARTIFACT_ROOT=$ART $PY -m j_pretrain.orchestration.run 2>&1 | tee logs/orch_\$(date -u +%Y%m%dT%H%M%SZ).log"

sleep 5
echo
echo "Started. Verify it grabbed the GPU and is resuming:"
echo "  tmux attach -t orch          # watch live (detach: Ctrl-b then d)"
echo "  nvidia-smi                   # expect ~16 GB / high util within a minute"
echo "  tail -f \$J_PRETRAIN_ARTIFACT_ROOT/run_metrics/music-300m_lambda-0.25__stage1.jsonl"
echo
echo "NOTE: this restarts TRAINING only. To restore full autonomy (health-checks,"
echo "bookkeeping, DAG advance narration) also restart the Ralph loop separately"
echo "(e.g. /ralph-loop), the way it was originally launched."
