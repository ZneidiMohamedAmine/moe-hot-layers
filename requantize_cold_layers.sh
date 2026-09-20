#!/bin/bash
# Hotness-aware mixed-precision requantization: keep the top-N hottest layers
# (by the same locality_sorted.tsv ranking pick_hot_layers.sh uses) at the
# model's original quant, and downgrade every other ("cold") layer's expert
# tensors to a smaller quant type. Cold layers are rarely-selected experts,
# so the quality cost is low while file size and disk-read cost drop a lot.
#
# Measured on Qwen3-30B-A3B, keeping the top 7 layers at Q4_K_M and dropping
# the other 41 to Q2_K: 18.56GB -> 11.61GB (-37%), and against the SAME
# hot-layer GPU placement: average generation speed 7.10 -> 8.71 tok/s
# (+23%), cold-start floor 2.20 -> 4.89 tok/s (+122%). Spot-checked output
# quality showed no visible degradation on code/story/explanation prompts.
set -e
cd "$(dirname "$0")"

INPUT="${1:?Usage: $0 <input.gguf> <output.gguf>}"
OUTPUT="${2:?Usage: $0 <input.gguf> <output.gguf>}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-llama-quantize}"

KEEP_HOT="${KEEP_HOT:-7}"              # how many top layers stay at full precision
COLD_QUANT="${COLD_QUANT:-Q2_K}"       # quant type for everything else
BASE_QUANT="${BASE_QUANT:-Q4_K_M}"     # the model's original quant type

TOTAL_LAYERS=$(wc -l < layer_sizes.tsv)
HOT_LAYERS=$(head -n "$KEEP_HOT" locality_sorted.tsv | awk '{print $1}')

echo "Total layers: $TOTAL_LAYERS" >&2
echo "Keeping top $KEEP_HOT hottest layers at $BASE_QUANT: $(echo $HOT_LAYERS | tr '\n' ' ')" >&2
echo "Downgrading the rest to $COLD_QUANT" >&2

OVERRIDE_FILE=$(mktemp)
trap 'rm -f "$OVERRIDE_FILE"' EXIT

for l in $(awk '{print $1}' layer_sizes.tsv); do
    if ! echo "$HOT_LAYERS" | grep -qw "$l"; then
        echo "blk.${l}.ffn_gate_exps.weight=${COLD_QUANT}" >> "$OVERRIDE_FILE"
        echo "blk.${l}.ffn_up_exps.weight=${COLD_QUANT}" >> "$OVERRIDE_FILE"
        echo "blk.${l}.ffn_down_exps.weight=${COLD_QUANT}" >> "$OVERRIDE_FILE"
    fi
done

echo "Requantizing (this dequantizes+requantizes the whole file, expect it to take a while)..." >&2
"$LLAMA_QUANTIZE" --allow-requantize --tensor-type-file "$OVERRIDE_FILE" "$INPUT" "$OUTPUT" "$BASE_QUANT"

echo "" >&2
echo "Done: $OUTPUT" >&2
echo "Use the same -ot string from pick_hot_layers.sh with this file -- layer/tensor names are unchanged, only cold-layer precision dropped." >&2
