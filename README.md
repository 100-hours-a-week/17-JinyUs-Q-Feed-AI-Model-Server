# QFeed GPU Server

GPU 인스턴스에서 STT 모델을 서빙하는 FastAPI 서버

## 환경 정보

- **인스턴스**: GCP Compute Engine (NVIDIA T4 (vCPU 4개, 메모리 16GB)) -> Runpod L4로 이전 예정
- **OS**: Ubuntu 
- **패키지 관리**: uv
- **프레임워크**: FastAPI + Uvicorn
- **python 버전** : Python 3.12

## 설치 및 실행

### 1. 프로젝트 초기화

```bash
uv sync
```

### 2. 서버 실행

로컬 실행

```bash
# 로컬 (기본값이라 생략 가능)
uv run uvicorn main:app --host 0.0.0.0 --port 8002

# 프로덕션
ENVIRONMENT=production uv run uvicorn main:app --host 0.0.0.0 --port 8002
```

### 3. 테스트 실행

```bash
uv run pytest
```

### Code Quality

```bash
uv run ruff check .
uv run ruff format --check .
```

## 파일 구조
``` bash
tree -I "node_modules|.git|**pycache**|audio_data|__pycache__|jupyter"
```

```bash
├── README.md
├── core
│   ├── logging.py
│   └── models.py
├── main.py
├── pyproject.toml
├── routers
│   └── stt.py
├── schemas
│   └── stt.py
└── uv.lock
```

### GPU Status
```bash
GET /gpu/status
```
Response:
```json
{
  "status": "ok",
  "gpus": [
    {
      "name": "NVIDIA L4",
      "memory_total_mb": 23034,
      "memory_used_mb": 0,
      "memory_free_mb": 23034,
      "utilization_percent": 0
    }
  ]
}
```
