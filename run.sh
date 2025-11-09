#!/usr/bin/env bash
set -euo pipefail

# Configurable via args: ./run_loop.sh [ITERATIONS] [SLEEP_SECS]
ITERATIONS=${1:-1} # default to 1 iteration 
SLEEP_SECS=${2:-1} # default to 1 second sleep

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python3

LOGDIR="$ROOT_DIR/logs"
mkdir -p "$LOGDIR"

for i in $(seq 1 "$ITERATIONS"); do
  echo "[$(date +%H:%M:%S)] Run $i/$ITERATIONS: generating airspace config..."
  "$PY" "$ROOT_DIR/airspace_config_gen.py"

  echo "[$(date +%H:%M:%S)] Run $i/$ITERATIONS: running rule-based evaluation..."
  ts=$(date +%Y%m%d_%H%M%S)
  "$PY" "$ROOT_DIR/RulebasedPolicyEvaluation_withLLMAgents.py" > "$LOGDIR/run_${i}_${ts}.log" 2>&1 || {
    echo "Evaluation exited non-zero (see $LOGDIR/run_${i}_${ts}.log)"
  }

  echo "[$(date +%H:%M:%S)] Completed run $i/$ITERATIONS"
  sleep "$SLEEP_SECS"
done

echo "All runs finished. Logs in: $LOGDIR"