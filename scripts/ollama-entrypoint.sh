#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-qwen3-vl:8b}"
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"

echo "Starting Ollama server on ${OLLAMA_HOST}..."
ollama serve &
OLLAMA_PID=$!

cleanup() {
  kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for Ollama API..."
for _ in $(seq 1 90); do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# Pull in the background so Azure readiness probes can pass while the
# (~5–10GB) multimodal weights download. Requests will 404 until ready.
echo "Ensuring model is available (background): ${MODEL}"
(
  set +e
  for attempt in 1 2 3 4 5; do
    echo "ollama pull attempt ${attempt}: ${MODEL}"
    if ollama pull "${MODEL}"; then
      echo "Ollama model ready: ${MODEL}"
      exit 0
    fi
    sleep 15
  done
  echo "ERROR: failed to pull ${MODEL}" >&2
  exit 1
) &

wait "$OLLAMA_PID"
