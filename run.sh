#!/bin/bash
# run.sh - vLLM 서버 실행 스크립트

set -e

# 설정
MODEL="skt/A.X-4.0-Light"
HOST="0.0.0.0"
PORT="8002"
GPU_UTIL="0.9"
MAX_MODEL_LEN="8192"

# conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh  # 경로 확인 필요
conda activate llm-server

echo "Starting vLLM server..."
echo "Model: ${MODEL}"
echo "Port: ${PORT}"

python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --host ${HOST} \
    --port ${PORT} \
    --gpu-memory-utilization ${GPU_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --trust-remote-code
