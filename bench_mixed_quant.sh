#!/bin/bash
# Compare original uniform Q4_K_M vs hotness-aware mixed quant (hot layers
# Q4_K_M, cold layers Q2_K) at the same hot-layer GPU placement.
set -e
cd "$(dirname "$0")"

ORIG_MODEL="${ORIG_MODEL:?Set ORIG_MODEL to the path of your original (uniform-quant) Qwen3-30B-A3B GGUF}"
MIXED_MODEL="${MIXED_MODEL:?Set MIXED_MODEL to the path of the output from requantize_cold_layers.sh}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
PORT="${PORT:-8092}"
REPS="${REPS:-3}"

OT=$(./pick_hot_layers.sh)
echo "OT string in use: $OT"

PROMPTS=(
  "Write a short function in Python that checks if a number is prime, then explain how it works."
  "Tell me a short story about a dragon who is afraid of heights."
  "Explain how photosynthesis works in simple terms."
)

run_config() {
    local label="$1"
    local model="$2"

    echo ""
    echo "===== $label ====="

    "$LLAMA_SERVER" -m "$model" -ngl 999 -ot "$OT" -c 4096 --port "$PORT" \
        > "server_${label// /_}.log" 2>&1 &
    local server_pid=$!

    echo "Waiting for server (pid $server_pid) to finish loading the model..."
    for i in $(seq 1 90); do
        resp=$(curl -s "http://127.0.0.1:${PORT}/completion" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"Hi","n_predict":1,"temperature":0}' 2>/dev/null)
        if echo "$resp" | grep -q '"content"'; then
            echo "Model ready after ~$((i*2))s"
            break
        fi
        sleep 2
    done

    local speeds=()
    for prompt in "${PROMPTS[@]}"; do
        for r in $(seq 1 $REPS); do
            resp=$(curl -s "http://127.0.0.1:${PORT}/completion" \
                -H "Content-Type: application/json" \
                -d "{\"prompt\": $(python -c "import json,sys; print(json.dumps(sys.argv[1]))" "$prompt"), \"n_predict\": 100, \"temperature\": 0}")
            tps=$(echo "$resp" | python -c "
import json,sys
try:
    d = json.load(sys.stdin)
    t = d.get('timings', {})
    print(round(t.get('predicted_per_second', 0), 2))
except Exception as e:
    print('ERR', e, file=sys.stderr)
    print(0)
")
            content=$(echo "$resp" | python -c "
import json,sys
try:
    d = json.load(sys.stdin)
    print(d.get('content','')[:120].replace(chr(10),' '))
except Exception:
    print('(parse error)')
")
            echo "  prompt=\"${prompt:0:40}...\" rep=$r -> ${tps} tok/s"
            echo "    sample: $content"
            speeds+=("$tps")
        done
    done

    echo "${speeds[@]}" | tr ' ' '\n' | python -c "
import sys
vals = [float(x) for x in sys.stdin.read().split() if x]
if vals:
    print(f'  AVERAGE: {sum(vals)/len(vals):.2f} tok/s')
    print(f'  MAX:     {max(vals):.2f} tok/s')
    print(f'  MIN:     {min(vals):.2f} tok/s')
"

    kill "$server_pid" 2>/dev/null || true
    sleep 3
    taskkill //F //IM llama-server.exe 2>/dev/null || true
    sleep 2
}

run_config "original_uniform_q4km" "$ORIG_MODEL"
run_config "mixed_hot_q4km_cold_q2k" "$MIXED_MODEL"

echo ""
echo "All configs done."
