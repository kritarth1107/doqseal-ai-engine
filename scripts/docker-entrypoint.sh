#!/usr/bin/env bash
# Keep extraction worker + API alive together. If either dies, exit so Azure
# Container Apps restarts the replica (avoids silent queue stalls).
set -euo pipefail

export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export WORKER_HEARTBEAT_FILE="${WORKER_HEARTBEAT_FILE:-/tmp/doqseal-worker-heartbeat}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-3031}"
WATCH_INTERVAL_SEC="${WORKER_WATCH_INTERVAL_SEC:-5}"
HEARTBEAT_STALE_SEC="${WORKER_HEARTBEAT_STALE_SEC:-180}"

cleanup() {
  if [[ -n "${WORKER_PID:-}" ]]; then kill "$WORKER_PID" 2>/dev/null || true; fi
  if [[ -n "${UVICORN_PID:-}" ]]; then kill "$UVICORN_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

echo "Starting extraction worker..."
python -m app.worker &
WORKER_PID=$!
# Seed heartbeat so health checks pass during cold start
date +%s >"$WORKER_HEARTBEAT_FILE"

echo "Starting API (uvicorn)..."
python -m uvicorn app.main:app --host "$HOST" --port "$PORT" &
UVICORN_PID=$!

echo "Supervisor watching worker_pid=$WORKER_PID uvicorn_pid=$UVICORN_PID"

while true; do
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "FATAL: extraction worker exited — restarting container" >&2
    exit 1
  fi
  if ! kill -0 "$UVICORN_PID" 2>/dev/null; then
    echo "FATAL: uvicorn exited — restarting container" >&2
    exit 1
  fi
  if [[ -f "$WORKER_HEARTBEAT_FILE" ]]; then
    now=$(date +%s)
    hb=$(cat "$WORKER_HEARTBEAT_FILE" 2>/dev/null || echo 0)
    age=$((now - hb))
    if (( age > HEARTBEAT_STALE_SEC )); then
      echo "FATAL: worker heartbeat stale (${age}s > ${HEARTBEAT_STALE_SEC}s) — restarting container" >&2
      exit 1
    fi
  fi
  sleep "$WATCH_INTERVAL_SEC"
done
