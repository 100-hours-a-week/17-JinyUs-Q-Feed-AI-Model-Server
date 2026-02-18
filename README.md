# QFeed vLLM Server

Q-feed 기술 면접 서비스를 위한 vLLM 기반 LLM 서빙 서버

## 환경 정보

- **인스턴스**: Runpod (GCP에서 이전)
- **GPU**: NVIDIA L4
- **모델**: `skt/A.X-4.0-Light`
- **conda 환경**: `llm-server`

## 설치

### 방법 1: environment.yml 사용 (권장)

```bash
conda env create -f environment.yml
conda activate llm-server
```

### 방법 2: 수동 설치

```bash
conda create -n llm-server python=3.12 -y
conda activate llm-server
pip install vllm torch
```

## 실행

### 로컬 실행

```bash
conda activate llm-server

python -m vllm.entrypoints.openai.api_server \
    --model skt/A.X-4.0-Light \
    --host 0.0.0.0 \
    --port 8002 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --trust-remote-code
```

또는 스크립트 사용:

```bash
./run.sh
```

### 프로덕션 실행 (systemd)



## 파일 구조

```
.
├── README.md           # 이 문서
├── run.sh              # 실행 스크립트
├── vllm.service        # systemd 서비스 설정
└── environment.yml     # conda 환경 설정
```

## 주요 설정 옵션

| 옵션 | 값 | 설명 |
|------|-----|------|
| `--gpu-memory-utilization` | 0.9 | GPU 메모리 사용률 (90%) |
| `--max-model-len` | 8192 | 최대 컨텍스트 길이 |
| `--trust-remote-code` | - | 모델의 커스텀 코드 허용 |