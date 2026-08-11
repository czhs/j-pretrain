#!/usr/bin/env bash
# restart_run.sh — un-shelve the j-pretrain experiment.
#
# The run is driven by the Ralph loop `loop.sh`, a detached shell that runs
# headless `claude -p` iterations. Each iteration reconstructs state and, if the
# orchestrator (GPU trainer) is not alive, relaunches it via the resume_cmd —
# which resume-detects the latest resumable checkpoint and continues byte-exact.
#
# So restarting the DRIVER is enough: it will bring the orchestrator (and thus
# GPU training) back automatically on its first iteration.
#
# Two modes:
#   ./restart_run.sh          # full autonomy: relaunch the Ralph driver (default)
#   ./restart_run.sh orch     # training only: relaunch just the orchestrator
set -euo pipefail

REPO="/home/hshi-j-4090/Desktop/j-pretrain"
ART="/home/hshi-j-4090/Desktop/j-pretrain-artifacts"
PY="/home/hshi-j-4090/miniconda3/envs/jpre/bin/python"
MODE="${1:-driver}"
cd "$REPO"

# --- guards ---------------------------------------------------------------------
if pgrep -af 'loop\.sh' | grep -q 'loop.sh'; then
  echo "ERROR: a Ralph driver (loop.sh) is already running:"; pgrep -af 'loop\.sh'
  echo "Refusing to start a second driver."; exit 1
fi
if tmux has-session -t orch 2>/dev/null; then
  echo "ERROR: tmux session 'orch' already exists (orchestrator running). Attach: tmux attach -t orch"
  exit 1
fi
if [ -f state/gpu.lock ]; then
  LOCKPID="$(grep -oE '"pid":[[:space:]]*[0-9]+' state/gpu.lock | grep -oE '[0-9]+' || true)"
  if [ -n "${LOCKPID:-}" ] && kill -0 "$LOCKPID" 2>/dev/null; then
    echo "ERROR: state/gpu.lock held by live pid $LOCKPID. Another job is running."; exit 1
  fi
  echo "Note: clearing stale state/gpu.lock."; rm -f state/gpu.lock
fi
USEDMB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "${USEDMB:-0}" -gt 4000 ]; then
  echo "WARNING: GPU already using ${USEDMB} MiB — someone may still be on it."
  read -r -p "Launch anyway? [y/N] " ans; [ "${ans:-N}" = "y" ] || { echo "Aborted."; exit 1; }
fi

if [ "$MODE" = "orch" ]; then
  # --- training only: resume the orchestrator (byte-exact) ----------------------
  echo "Relaunching orchestrator only (no autonomy) in tmux 'orch'..."
  tmux new-session -d -s orch \
    "HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false J_PRETRAIN_ARTIFACT_ROOT=$ART $PY -m j_pretrain.orchestration.run 2>&1 | tee logs/orch_\$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "Started. Watch: tmux attach -t orch   |   nvidia-smi"
  exit 0
fi

# --- full autonomy: relaunch the Ralph driver ----------------------------------
# Re-arm the plugin flag and start loop.sh in its own tmux session so it is easy
# to find and stop next time (tmux attach -t ralph / tmux kill-session -t ralph).
sed -i 's/^active: false/active: true/' .claude/ralph-loop.local.md 2>/dev/null || true
echo "Relaunching Ralph driver 'loop.sh 800' in tmux session 'ralph'..."
tmux new-session -d -s ralph "cd $REPO && exec bash ./loop.sh 800"
sleep 3
echo
echo "Started. The driver will resume the orchestrator (byte-exact) on its first"
echo "iteration — expect GPU to fill within a minute or two."
echo "  tmux attach -t ralph     # watch the loop  (detach: Ctrl-b then d)"
echo "  tmux attach -t orch      # watch training once it relaunches"
echo "  nvidia-smi               # expect ~16 GB / high util shortly"
echo
echo "To shelve again: tmux kill-session -t ralph && tmux kill-session -t orch && rm -f state/gpu.lock"
