#!/bin/bash
# run.sh - vLLM 서버 실행 스크립트

set -e

# 설정
MODEL="skt/A.X-4.0-Light"
HOST="0.0.0.0"
PORT="8002"
GPU_UTIL="0.9"
MAX_MODEL_LEN="9500"

echo "Starting vLLM server..."
echo "Model: ${MODEL}"
echo "Port: ${PORT}"

uv run python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --host ${HOST} \
    --port ${PORT} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --trust-remote-code \
    --enable-prefix-caching
