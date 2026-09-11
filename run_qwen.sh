#!/bin/bash
# Launches Qwen3-30B-A3B via llama-cli, with hot MoE layers auto-placed on GPU
# (see pick_hot_layers.sh). Extra args are passed through to llama-cli. Run
# with no args for a real interactive chat session; pass -p "..." for a
# scripted one-shot answer.
set -e
cd "$(dirname "$0")"

# Point these at your own setup.
MODEL="${MODEL:?Set MODEL to the path of your Qwen3-30B-A3B GGUF file (any Q4_K_M-class quant of the base or a compatible finetune)}"
LLAMA_CLI="${LLAMA_CLI:-llama-cli}"   # override if it's not on your PATH

OT=$(./pick_hot_layers.sh)

exec "$LLAMA_CLI" \
    -m "$MODEL" \
    -ngl 999 \
    -ot "$OT" \
    "$@"
