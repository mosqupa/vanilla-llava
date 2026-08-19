#!/bin/bash

set -e
export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_PYTHON="/opt/conda/envs/llava/bin/python"

SPLIT="${SPLIT:-refcoco_val_questions}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/llava-v1.5-7b}"
MODEL_NAME="${MODEL_NAME:-llava-v1.5-7b}"
DATA_DIR="$PROJECT_ROOT/data/refcoco"
PE_FLAG=""
if [ "${USE_2D_PE:-true}" = "true" ] || [ "${USE_2D_PE:-true}" = "1" ]; then
    PE_FLAG="--use-2d-pe"
fi

echo "========================================="
echo "  RefCOCO — $SPLIT"
echo "  Model: $MODEL_PATH"
echo "========================================="

# Step 0 — Convert data (idempotent)
$CONDA_PYTHON "$PROJECT_ROOT/scripts/refcoco_converter.py" \
    --splits-dir "$DATA_DIR" \
    --output-dir "$DATA_DIR/converted" \
    --splits "${SPLIT%.json}"

# Step 1+2 — Inference + Evaluation
$CONDA_PYTHON "$PROJECT_ROOT/scripts/refcoco_inference.py" \
    --split "$SPLIT" \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    --data-dir "$DATA_DIR" \
    --pruning-method "${PRUNING_METHOD:-random}" \
    --keep-ratio "${KEEP_RATIO:-0.125}" \
    $PE_FLAG \
    --pe-scale "${PE_SCALE:-1.0}" \
