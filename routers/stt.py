# routers/stt.py
"""Speech-to-Text 라우터"""

import io
import time

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydub import AudioSegment

from core.models import model_manager, get_logger
from schemas.stt import STTResponse

router = APIRouter(prefix="/whisper", tags=["STT"])
logger = get_logger(__name__)

# 언어 코드 매핑
LANGUAGE_MAP = {
    "ko": "korean",
    "en": "english",
    "ja": "japanese",
    "zh": "chinese",
}
BATCH_SIZE = 8


def decode_audio(audio_data: bytes, filename: str | None) -> tuple[np.ndarray, float]:
    """
    오디오 데이터를 numpy array로 디코딩
    
    Returns:
        (audio_array, load_time)
    """
    load_start = time.perf_counter()
    
    # 파일 확장자 추출
    fmt = filename.split('.')[-1] if filename else "mp3"
    
    # pydub로 디코딩
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=fmt)
    
    # 16kHz 모노로 변환
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    
    # numpy array로 변환
    samples = np.array(audio_segment.get_array_of_samples())
    
    # float32로 정규화
    if audio_segment.sample_width == 2:  # 16-bit
        audio_array = samples.astype(np.float32) / 32768.0
    elif audio_segment.sample_width == 4:  # 32-bit
        audio_array = samples.astype(np.float32) / 2147483648.0
    else:  # 8-bit
        audio_array = (samples.astype(np.float32) - 128) / 128.0
    
    # 1D로 평탄화
    if audio_array.ndim != 1:
        audio_array = audio_array.flatten()
    
    load_time = time.perf_counter() - load_start
    return audio_array, load_time


@router.post("/stt", response_model=STTResponse)
async def transcribe(
    audio: UploadFile = File(..., description="오디오 파일 (mp3, mp4, wav 등)"),
    language: str = Form("ko", description="언어 코드 (ko, en, ja, zh)")
):
    """
    Whisper Turbo를 사용한 음성 인식
    """
    if not model_manager.is_whisper_ready:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    start_time = time.perf_counter()
    
    # 1. 오디오 데이터 읽기
    audio_data = await audio.read()
    read_time = time.perf_counter() - start_time
    
    # 2. 오디오 디코딩
    try:
        audio_array, decode_time = decode_audio(audio_data, audio.filename)
    except Exception as e:
        logger.error(f"Audio decoding failed | filename={audio.filename} | {e}")
        raise HTTPException(status_code=400, detail=f"Audio decoding failed: {e}")
    
    # 3. STT 수행
    try:
        stt_start = time.perf_counter()
        
        language_name = LANGUAGE_MAP.get(language, language)
        
        result = model_manager.whisper_pipe(
            {"raw": audio_array, "sampling_rate": 16000},
            generate_kwargs={
                "language": language_name,
                "task": "transcribe",
            },
            chunk_length_s=30,
            batch_size=BATCH_SIZE,
        )
        
        text = result["text"].strip()
        stt_time = time.perf_counter() - stt_start
        
    except Exception as e:
        logger.error(f"STT 처리 실패 | language={language} | error={e}")
        raise HTTPException(status_code=500, detail=f"STT error: {e}")
    
    # 결과 계산
    total_time = time.perf_counter() - start_time
    duration = len(audio_array) / 16000
    
    logger.info(
        f"STT(turbo) | read={read_time*1000:.0f}ms | "
        f"decode={decode_time*1000:.0f}ms | stt={stt_time*1000:.0f}ms | "
        f"total={total_time*1000:.0f}ms | duration={duration:.1f}s"
    )    
    return STTResponse(
        text=text,
        language=language,
        duration=duration,
        processing_time_ms=total_time * 1000,
    )
