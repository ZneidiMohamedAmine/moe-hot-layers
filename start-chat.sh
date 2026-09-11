#!/bin/bash
# Starts llama-server with hot MoE layers auto-placed on GPU (see
# pick_hot_layers.sh), then opens llama.cpp's built-in web chat UI in your
# browser. No extra frontend needed -- llama-server ships one.
set -e
cd "$(dirname "$0")"

MODEL="${MODEL:?Set MODEL to the path of your Qwen3-30B-A3B GGUF file}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"   # override if it's not on your PATH
PORT="${PORT:-8080}"

OT=$(./pick_hot_layers.sh)

"$LLAMA_SERVER" \
    -m "$MODEL" \
    -ngl 999 \
    -ot "$OT" \
    -c 8192 \
    --port "$PORT" \
    "$@" &

echo "Waiting for the server to come up..."
sleep 8

if command -v xdg-open &>/dev/null; then
    xdg-open "http://127.0.0.1:${PORT}"
elif command -v open &>/dev/null; then
    open "http://127.0.0.1:${PORT}"
else
    start "http://127.0.0.1:${PORT}" 2>/dev/null || echo "Open http://127.0.0.1:${PORT} in your browser."
fi

wait
