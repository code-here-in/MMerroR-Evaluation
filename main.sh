#!/usr/bin/env sh
set -eu

(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

CONFIG="${CONFIG:-eval_config.yaml}"
API_BASE="${MMERROR_API_BASE:-${API_BASE:-}}"
API_KEY="${MMERROR_API_KEY:-${API_KEY:-}}"
DATA_DIR="${DATA_DIR:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MODELS="${MODELS:-}"
TASKS="${TASKS:-}"

set -- python auto_eval_all_models.py --config "${CONFIG}"

if [ -n "${DATA_DIR}" ]; then
    set -- "$@" --data-dir "${DATA_DIR}"
fi
if [ -n "${IMAGE_DIR}" ]; then
    set -- "$@" --image-dir "${IMAGE_DIR}"
fi
if [ -n "${OUTPUT_DIR}" ]; then
    set -- "$@" --output-dir "${OUTPUT_DIR}"
fi
if [ -n "${API_BASE}" ]; then
    set -- "$@" --api-base "${API_BASE}"
fi
if [ -n "${API_KEY}" ]; then
    set -- "$@" --key "${API_KEY}"
fi
if [ -n "${MODELS}" ]; then
    set -- "$@" --models "${MODELS}"
fi
if [ -n "${TASKS}" ]; then
    set -- "$@" --tasks "${TASKS}"
fi

"$@"
