#!/usr/bin/env bash
# =============================================================================
# Parallel inference + scoring across 10 GPUs
#
# Usage (HF backend — start servers first with start_hf_servers.sh):
#
#   bash scripts_parallel/run_eval.sh hf qwen3_omni
#   bash scripts_parallel/run_eval.sh hf qwen3_vl
#
# Usage (Gemini cloud API — no local server needed):
#
#   bash scripts_parallel/run_eval.sh gemini gemini_3_pro
#
# Score only (skip inference):
#
#   bash scripts_parallel/run_eval.sh score qwen3_omni /path/to/output.jsonl
#
# Custom options:
#   NUM_WORKERS=5 VIDEO_ROOT=/data/videos bash scripts_parallel/run_eval.sh hf qwen3_omni
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:?Usage: bash scripts_parallel/run_eval.sh <hf|gemini|score> <model_name> [model_output]}"
MODEL_NAME="${2:?Please specify model_name, e.g. qwen3_omni / qwen3_vl / gemini_3_pro}"
MODEL_OUTPUT="${3:-}"

# ===================== Configurable =====================
NUM_WORKERS="${NUM_WORKERS:-10}"
NUM_SCORERS="${NUM_SCORERS:-10}"
NUM_GPUS="${NUM_GPUS:-10}"
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
VIDEO_ROOT="${VIDEO_ROOT:-${SCRIPT_DIR}/videos}"
STREAM_ADDR_ROOT="${STREAM_ADDR_ROOT:-}"
CHUNK_CACHE_ROOT="${CHUNK_CACHE_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# ========================================================

build_api_bases() {
    local bases=""
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        local port=$((BASE_PORT + i))
        if [[ -n "$bases" ]]; then bases="${bases},"; fi
        bases="${bases}http://${HOST}:${port}/v1"
    done
    echo "$bases"
}

common_args=(
    --model-name "$MODEL_NAME"
    --output-dir "$OUTPUT_DIR"
    --run-id "$RUN_ID"
    --num-scorers "$NUM_SCORERS"
)

if [[ -n "$VIDEO_ROOT" ]]; then
    common_args+=(--video-root "$VIDEO_ROOT")
fi
if [[ -n "$STREAM_ADDR_ROOT" ]]; then
    common_args+=(--stream-addr-root "$STREAM_ADDR_ROOT")
fi
if [[ -n "$CHUNK_CACHE_ROOT" ]]; then
    common_args+=(--chunk-cache-root "$CHUNK_CACHE_ROOT")
fi

echo "=============================================="
echo " StreamEval Parallel Evaluation"
echo "=============================================="
echo " MODE        : $MODE"
echo " MODEL_NAME  : $MODEL_NAME"
echo " NUM_WORKERS : $NUM_WORKERS"
echo " NUM_SCORERS : $NUM_SCORERS"
echo " OUTPUT_DIR  : $OUTPUT_DIR"
echo " RUN_ID      : $RUN_ID"
echo "=============================================="

case "$MODE" in
    hf)
        API_BASES=$(build_api_bases)
        echo " API_BASES   : $API_BASES"
        echo "=============================================="
        echo ""
        python run_stream_eval_parallel.py \
            "${common_args[@]}" \
            --num-workers "$NUM_WORKERS" \
            --api-bases "$API_BASES" \
            $EXTRA_ARGS
        ;;

    gemini)
        echo " (Gemini cloud API — no local server needed)"
        echo "=============================================="
        echo ""
        python run_stream_eval_parallel.py \
            "${common_args[@]}" \
            --num-workers "$NUM_WORKERS" \
            $EXTRA_ARGS
        ;;

    score)
        if [[ -z "$MODEL_OUTPUT" ]]; then
            echo "[ERROR] score mode requires a model_output path"
            echo "  Usage: bash scripts_parallel/run_eval.sh score <model_name> <model_output_path>"
            exit 1
        fi
        echo " MODEL_OUTPUT: $MODEL_OUTPUT"
        echo "=============================================="
        echo ""
        python run_stream_eval_parallel.py \
            "${common_args[@]}" \
            --skip-inference \
            --model-output "$MODEL_OUTPUT" \
            $EXTRA_ARGS
        ;;

    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo "  Supported: hf / gemini / score"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo " Done! Results saved to: ${OUTPUT_DIR}/${MODEL_NAME}_${RUN_ID}/"
echo "=============================================="
