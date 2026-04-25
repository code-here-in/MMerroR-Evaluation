#!/usr/bin/env sh
set -eu

(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

API_BASE="${API_BASE:-${MMERROR_API_BASE:-}}"
API_KEY="${API_KEY:-${MMERROR_API_KEY:-}}"
DATA_DIR="${DATA_DIR:-../data/jsons}"
IMAGE_DIR="${IMAGE_DIR:-../data/images}"
OUTPUT_DIR="${OUTPUT_DIR:-../result}"

if [ -z "${API_KEY}" ]; then
    echo "[ERROR] API_KEY/MMERROR_API_KEY is empty."
    echo "Example:"
    echo "  API_BASE=\"https://your-endpoint/v1/chat/completions\" API_KEY=\"your-key\" sh main.sh"
    exit 2
fi

python auto_eval_all_models.py \
    --data-dir "${DATA_DIR}" \
    --image-dir "${IMAGE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --api-base "${API_BASE}" \
    --key "${API_KEY}" \
    --models "gpt-5.2,claude-opus-4.5,gemini-3-pro-preview" \
    --tasks "etc,epd"

