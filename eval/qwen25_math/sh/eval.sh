#!/usr/bin/env bash
set -euo pipefail

PROMPT_TYPE="qwen25-math-cot"
DATA_NAME="math"
PRED_FILE="outputs/preds.jsonl"

python evaluate.py \
  --data_name "$DATA_NAME" \
  --prompt_type "$PROMPT_TYPE" \
  --file_path "$PRED_FILE"
