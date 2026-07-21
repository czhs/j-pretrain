#!/bin/bash
set -u
PROMPT_FILE="REPRO_PROMPT.md"
PROMISE="<promise>EXPERIMENTS_COMPLETE</promise>"
LOG_DIR="logs/ralph"
LIMIT_SLEEP=1800   # 30 min between retries while usage-limited
ITER_SLEEP=15      # small breather between normal iterations

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi
if [ ! -s "$PROMPT_FILE" ]; then
  echo "ERROR: $PROMPT_FILE missing or empty" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

i=0
while [ "$i" -lt "$1" ]; do
  i=$((i+1))
  ts=$(date +%Y%m%d_%H%M%S)
  log="$LOG_DIR/iter_${i}_${ts}.log"
  echo "=== Iteration $i ($ts) -> $log ==="

  claude -p "$(cat "$PROMPT_FILE")" --output-format text --dangerously-skip-permissions 2>&1 | tee "$log"
  echo "--- end iteration $i (exit ${PIPESTATUS[0]}) ---"

  # Usage/rate limit: sleep and don't count the attempt against the budget
  if grep -qiE 'usage limit|rate limit|limit reached|resets at' "$log"; then
    echo "Usage limit detected; sleeping ${LIMIT_SLEEP}s before retry..."
    i=$((i-1))
    sleep "$LIMIT_SLEEP"
    continue
  fi

  # Promise emitted: trust but verify via the completion verifier
  if grep -qF "$PROMISE" "$log"; then
    if [ -f scripts/verify_completion.py ] && python scripts/verify_completion.py; then
      echo "Completion promise confirmed by verifier after $i iterations."
      exit 0
    else
      echo "WARNING: promise emitted but verifier missing or failing; continuing."
    fi
  fi

  sleep "$ITER_SLEEP"
done
echo "Reached max iterations ($1)"
exit 1
