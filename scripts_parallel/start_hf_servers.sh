#!/usr/bin/env bash
# =============================================================================
# Launch one hf_openai_server instance per GPU
#
# Usage:
#   bash scripts_parallel/start_hf_servers.sh            # use default config
#   MODEL_PATH=/data/Qwen3-VL-8B  bash scripts_parallel/start_hf_servers.sh
#   NUM_GPUS=4 BASE_PORT=9000      bash scripts_parallel/start_hf_servers.sh
#
# Stop all servers:
#   bash scripts_parallel/start_hf_servers.sh stop
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ===================== Configurable =====================
MODEL_PATH="${MODEL_PATH:-/path/to/your/HF_MODEL}"
MODEL_TYPE="${MODEL_TYPE:-auto}"                   # auto | qwen3_omni | qwen3_vl | text
TORCH_DTYPE="${TORCH_DTYPE:-auto}"                 # auto | bf16 | fp16
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
USE_AUDIO="${USE_AUDIO:-1}"                        # 1=true, 0=false (qwen3_omni only)
NUM_GPUS="${NUM_GPUS:-10}"
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/servers}"
# ========================================================

mkdir -p "$LOG_DIR"
PID_FILE="${LOG_DIR}/server_pids.txt"

stop_servers() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "[INFO] No PID file found at $PID_FILE, nothing to stop."
        return
    fi
    echo "[INFO] Stopping servers ..."
    while IFS= read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "  killed PID $pid" || true
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    echo "[INFO] All servers stopped."
}

if [[ "${1:-}" == "stop" ]]; then
    stop_servers
    exit 0
fi

# Clean up leftover processes from previous runs
stop_servers 2>/dev/null || true
> "$PID_FILE"

echo "=============================================="
echo " StreamEval HF Model Server Launcher"
echo "=============================================="
echo " MODEL_PATH  : $MODEL_PATH"
echo " MODEL_TYPE  : $MODEL_TYPE"
echo " NUM_GPUS    : $NUM_GPUS"
echo " BASE_PORT   : $BASE_PORT"
echo " TORCH_DTYPE : $TORCH_DTYPE"
echo " ATTN_IMPL   : $ATTN_IMPL"
echo " LOG_DIR     : $LOG_DIR"
echo "=============================================="

for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + i))
    log_file="${LOG_DIR}/gpu${i}_port${port}.log"

    echo "[GPU $i] Starting on port $port ..."
    CUDA_VISIBLE_DEVICES="$i" python hf_openai_server.py \
        --model-path "$MODEL_PATH" \
        --model-type "$MODEL_TYPE" \
        --host "$HOST" \
        --port "$port" \
        --device-map auto \
        --torch-dtype "$TORCH_DTYPE" \
        --attn-implementation "$ATTN_IMPL" \
        --use-audio-in-video "$USE_AUDIO" \
        > "$log_file" 2>&1 &

    echo $! >> "$PID_FILE"
done

echo ""
echo "[INFO] Waiting for all servers to become healthy ..."
all_healthy=true
for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + i))
    healthy=false
    for attempt in $(seq 1 120); do
        if curl -s "http://${HOST}:${port}/health" | grep -q '"ok"' 2>/dev/null; then
            echo "  [GPU $i] port $port  ✓  (${attempt}s)"
            healthy=true
            break
        fi
        sleep 1
    done
    if [[ "$healthy" == "false" ]]; then
        echo "  [GPU $i] port $port  ✗  TIMEOUT after 120s"
        echo "           Check log: ${LOG_DIR}/gpu${i}_port${port}.log"
        all_healthy=false
    fi
done

echo ""
if [[ "$all_healthy" == "true" ]]; then
    echo "=============================================="
    echo " All $NUM_GPUS servers are healthy!"
    echo "=============================================="
else
    echo "[WARN] Some servers failed to start. Check logs in $LOG_DIR"
fi

echo ""
echo "API bases for --api-bases:"
bases=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + i))
    if [[ -n "$bases" ]]; then bases="${bases},"; fi
    bases="${bases}http://${HOST}:${port}/v1"
done
echo "  $bases"
echo ""
echo "To stop all servers:  bash scripts_parallel/start_hf_servers.sh stop"
echo "PIDs saved to: $PID_FILE"
