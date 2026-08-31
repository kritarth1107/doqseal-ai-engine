#!/usr/bin/env bash
set -euo pipefail

# Avoid torch/transformers dynamo clash that breaks VLM + sentence-transformers
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

# Run extraction worker in background; FastAPI health/chat in foreground
python -m app.worker &
WORKER_PID=$!

cleanup() {
  kill "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec python -m uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-3031}"
