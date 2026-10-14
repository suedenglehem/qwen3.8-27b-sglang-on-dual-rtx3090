#!/usr/bin/env bash
#
# run_i7.sh — run the Qwen3.8-27B server on THIS host (the i7: 2x ~16 GB Ada cards).
#
# The qwen38 preset hardcodes --context-length 262144, which does NOT fit in 16 GB.
# This helper pins the i7-correct args: INT4 + port 8080 + a 140k context cap.
# Verified on this host: weights ~8.8 GB/card, KV pool ~215k tokens, ~2.4 GB
# free/card, decode CUDA graph captured, server healthy on :8080.
#
# Usage (from /dd2/sglang on the i7):
#   ./run_i7.sh            # start (detached)
#   ./run_i7.sh status     # running? GPU mem? health?
#   ./run_i7.sh stop       # stop it
#
# Push the context cap higher (the KV pool is ~215k) with:  CTX=180000 ./run_i7.sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="${CTX:-140000}"
PORT="${PORT:-8080}"
LOG="$SCRIPT_DIR/server-qwen38.log"
PAT="sglang[.]launch_server"   # bracket trick: matches the server, not this script

case "${1:-start}" in
  start)
    if pids="$(pgrep -f "$PAT")"; then
      echo "already running: $(echo "$pids" | tr '\n' ' ')"
      exit 0
    fi
    echo "starting: int4, port=$PORT, context=$CTX (pool ~215k)"
    setsid nohup "$SCRIPT_DIR/run_qwen38.sh" -i4 --port "$PORT" --context-length "$CTX" \
      </dev/null >"$LOG" 2>&1 &
    echo "launched (detached). log: $LOG"
    echo "  watch : tail -f $LOG"
    echo "  status: $0 status"
    ;;
  status)
    if pids="$(pgrep -f "$PAT")"; then
      echo "running: $(echo "$pids" | tr '\n' ' ')"
      echo "--- GPU ---"
      nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
      echo "--- health ---"
      curl -s "http://127.0.0.1:$PORT/health" || echo "(no response — still booting?)"
      echo
    else
      echo "not running"
    fi
    ;;
  stop)
    if pids="$(pgrep -f "$PAT")"; then
      mainpid="$(echo "$pids" | head -1)"
      pgid="$(ps -o pgid= -p "$mainpid" | tr -d ' ')"
      if kill -- -"$pgid" 2>/dev/null; then
        echo "SIGTERM to process group $pgid"
        echo "  confirm: $0 status   (GPU should drop to ~0 MiB)"
      else
        echo "could not signal group $pgid (already gone?)"
      fi
    else
      echo "not running"
    fi
    ;;
  *)
    echo "usage: $0 [start|status|stop]" >&2
    exit 1
    ;;
esac
