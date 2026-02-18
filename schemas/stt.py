# schemas/stt.py
from pydantic import BaseModel

class STTResponse(BaseModel):
    text: str
    language: str
    duration: float
    processing_time_ms: float
