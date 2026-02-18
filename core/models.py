# core/models.py
"""GPU 모델 로딩 및 관리"""

import torch
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from core.logging import get_logger

# 설정
WHISPER_TURBO_MODEL_ID = "openai/whisper-large-v3-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

logger = get_logger(__name__)


class ModelManager:
    """GPU 모델들을 관리하는 싱글톤 클래스"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.whisper_pipe = None
        self._initialized = True
    
    def load_whisper(self) -> float:
        """Whisper 모델 로드. 소요 시간(초) 반환"""
        
        
        logger.info(f"Loading whisper turbo model: {WHISPER_TURBO_MODEL_ID}...")
        start = time.perf_counter()
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_TURBO_MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(DEVICE)
        
        processor = AutoProcessor.from_pretrained(WHISPER_TURBO_MODEL_ID)
        
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
        )
        
        elapsed = time.perf_counter() - start
        logger.info(f"Whisper turbo model loaded in {elapsed:.2f}s")
        return elapsed
    
    def unload_all(self):
        """모든 모델 언로드"""
        self.whisper_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Models unloaded")
    
    @property
    def is_whisper_ready(self) -> bool:
        return self.whisper_pipe is not None


# 전역 인스턴스
model_manager = ModelManager()
