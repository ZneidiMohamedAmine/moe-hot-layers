#!/bin/bash
# Greedily fills the free VRAM budget with the hottest MoE layers (by expert-reuse
# locality) and emits an -ot override string for llama.cpp. Reuses data already
# gathered this session: locality_sorted.tsv (layer, avg_overlap desc) and
# layer_sizes.tsv (layer, expert-tensor MiB).
set -e
cd "$(dirname "$0")"

RESERVE_MIB=${RESERVE_MIB:-1200}   # shared weights + KV + compute buffer + safety margin
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
BUDGET=$((FREE_MIB - RESERVE_MIB))

echo "free VRAM: ${FREE_MIB} MiB, reserved: ${RESERVE_MIB} MiB, budget for hot layers: ${BUDGET} MiB" >&2

declare -A SIZE
while read -r layer mib; do SIZE[$layer]=$mib; done < layer_sizes.tsv

layers=()
used=0
while read -r layer overlap; do
    sz=${SIZE[$layer]}
    if [ $((used + sz)) -le "$BUDGET" ]; then
        layers+=("$layer")
        used=$((used + sz))
    fi
done < locality_sorted.tsv

echo "selected ${#layers[@]} layers (${used} MiB): ${layers[*]}" >&2

pattern=$(IFS='|'; echo "${layers[*]}")
echo "blk\\.(${pattern})\\.ffn_(gate|up|down)_exps\\.weight=CUDA0,blk\\..*\\.ffn_.*_exps\\.weight=CPU"
