#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_background.sh — Run the full vLLM benchmark suite in the background.
#
# • Automatically resumes from the most-recent partial run (--resume).
# • All output is tee'd to a timestamped log file.
# • Writes a PID file so you can stop the run easily.
# • Safe to call again if a run is already active (exits with a warning).
#
# Usage:
#   ./run_background.sh                          # full suite, auto-resume
#   ./run_background.sh --models openai/gpt-oss-20b
#   ./run_background.sh --tp 4 8 --quantization none fp8
#   ./run_background.sh --no-resume              # force a fresh run
#
# Monitoring:
#   tail -f <log file printed on start>
#   ./experiment_utils.py status
#   ./experiment_utils.py logs
#
# Stopping:
#   kill $(cat experiment_bg.pid)
#   ./experiment_utils.py stop
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/experiment_bg.pid"

# ---- parse --no-resume flag (all other args passed through) ----
RESUME_FLAG="--resume"
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-resume" ]]; then
        RESUME_FLAG=""
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

# ---- guard against double-launch ----
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: An experiment is already running (PID $OLD_PID)."
        echo "  To stop it : kill $OLD_PID"
        echo "  Or         : ./experiment_utils.py stop && kill $OLD_PID"
        exit 1
    else
        echo "Stale PID file found (PID $OLD_PID no longer running); removing."
        rm -f "$PID_FILE"
    fi
fi

LOG_FILE="$SCRIPT_DIR/bg_run_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo " vLLM Benchmark — Background Launcher"
echo "========================================"
echo "Log    : $LOG_FILE"
echo "PID    : $PID_FILE"
[[ -n "$RESUME_FLAG" ]] && echo "Mode   : resume (skip already-completed experiments)" \
                        || echo "Mode   : fresh run"
[[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]] && echo "Args   : ${PASSTHROUGH_ARGS[*]}"
echo "========================================"
echo ""

# ---- launch detached ----
# shellcheck disable=SC2086
nohup "$SCRIPT_DIR/run_experiments.py" $RESUME_FLAG "${PASSTHROUGH_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
BG_PID=$!
echo "$BG_PID" > "$PID_FILE"

echo "Started (PID $BG_PID)"
echo ""
echo "Commands:"
echo "  Monitor  : tail -f $LOG_FILE"
echo "  Status   : ./experiment_utils.py status"
echo "  Stop run : kill \$(cat $PID_FILE)"
echo ""
echo "The terminal is now free. The run continues in the background."
