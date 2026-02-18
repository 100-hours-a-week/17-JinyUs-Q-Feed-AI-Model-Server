# main.py
"""QFeed GPU Server - FastAPI Application"""

import subprocess
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.models import model_manager
from core.logging import setup_logging, RequestLoggingMiddleware, get_logger
from routers.stt import router as stt_router

# 환경 설정
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

setup_logging(environment=ENVIRONMENT)
logger = get_logger(__name__)   

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드, 종료 시 정리"""
    logger.info(f"server start | env={ENVIRONMENT}")

    # Startup
    model_manager.load_whisper()
    
    yield
    
    # Shutdown
    model_manager.unload_all()
    logger.info("server shutdown")


app = FastAPI(
    title="QFeed GPU Server",
    description="GPU 기반 AI 모델 서빙 서버 (STT, TTS 등)",
    lifespan=lifespan,
)

# 미들웨어 등록
app.add_middleware(RequestLoggingMiddleware)

# 라우터 등록
app.include_router(stt_router)


@app.get("/health", tags=["Health"])
def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


@app.get("/gpu/status", tags=["Health"])
def gpu_status():
    """GPU 상태 확인"""
    try:
        result = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            name, mem_total, mem_used, mem_free, util = [x.strip() for x in line.split(',')]
            gpus.append({
                "name": name,
                "memory_total_mb": int(mem_total),
                "memory_used_mb": int(mem_used),
                "memory_free_mb": int(mem_free),
                "utilization_percent": int(util),
            })
        
        return {"status": "ok", "gpus": gpus}
    
    except FileNotFoundError:
        return {"status": "error", "message": "nvidia-smi not found"}
    except Exception as e:
        logger.warning(f"GPU status check failed | error={e}")
        return {"status": "error", "message": str(e)}


@app.get("/models/status", tags=["Health"])
def models_status():
    """로드된 모델 상태 확인"""
    return {
        "whisper": model_manager.is_whisper_ready,
    }
