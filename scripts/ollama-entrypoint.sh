#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-gemma4:e4b}"
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

echo "Ensuring model is available: ${MODEL}"
ollama pull "${MODEL}"

echo "Ollama ready with model ${MODEL}"
wait "$OLLAMA_PID"
