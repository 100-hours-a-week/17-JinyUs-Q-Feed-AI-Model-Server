# QFeed vLLM Server

Q-feed 기술 면접 서비스를 위한 vLLM 기반 LLM 서빙 서버

## 환경 정보

- **인스턴스**: Runpod (GCP에서 이전)
- **GPU**: NVIDIA L4
- **모델**: `openai/gpt-oss-20b`
- **패키지 매니저**: uv

## 사전 요구사항

- CUDA 12.x 이상 (Docker CUDA 베이스 이미지 또는 시스템 설치)
- [uv](https://docs.astral.sh/uv/) 설치:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## 설치

```bash
uv sync
```

## 실행

### 스크립트 실행 (권장)

```bash
./run.sh
```

### 수동 실행

```bash
uv run python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8002 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 9500 \
    --trust-remote-code \
    --enable-prefix-caching
```

### Docker 실행

```bash
docker build -t qfeed-vllm .
docker run --gpus all -p 8002:8002 qfeed-vllm
```

## 파일 구조

```
.
├── README.md           # 이 문서
├── pyproject.toml      # 프로젝트 설정 및 의존성
├── run.sh              # 실행 스크립트
└── requirements.txt    # Python 패키지 의존성 (호환용)
```

## 주요 설정 옵션

| 옵션 | 값 | 설명 |
|------|-----|------|
| `--gpu-memory-utilization` | 0.9 | GPU 메모리 사용률 (90%) |
| `--max-model-len` | 8192 | 최대 컨텍스트 길이 |
| `--trust-remote-code` | - | 모델의 커스텀 코드 허용 |