#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

pip install -r requirements.txt

mkdir -p "$(dirname "$OUTPUT_PATH")"

python predict.py \
    --data-dir "$DATA_DIR" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"