"""Pydantic schemas for ASR API responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WordTimestampSchema(BaseModel):
    text: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")


class SegmentSchema(BaseModel):
    id: int
    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds")
    text: str
    words: Optional[List[WordTimestampSchema]] = None


class TranscriptionResultSchema(BaseModel):
    model: str
    language: str
    duration_sec: float
    processing_time_sec: float
    rtf: float = Field(default=0.0, description="Real-Time Factor (processing_time / audio_duration)")
    text: str
    segments: List[SegmentSchema] = Field(default_factory=list)
    words: Optional[List[WordTimestampSchema]] = None


class TaskStatusSchema(BaseModel):
    task_id: str
    status: str = Field(description="queued | decoding | processing | completed | failed")
    filename: Optional[str] = None
    duration_sec: Optional[float] = None
    total_chunks: int = 0
    completed_chunks: int = 0
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    language_requested: Optional[str] = None
    return_timestamps: bool = True
    include_words: bool = True
    created_at: float
    updated_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[TranscriptionResultSchema] = None


class WorkerStatusSchema(BaseModel):
    gpu_id: str
    pid: Optional[int] = None
    alive: bool


class HealthResponseSchema(BaseModel):
    status: str
    backend: str
    model: str
    aligner_enabled: bool
    workers: List[WorkerStatusSchema]
    inflight_batches: int
    active_background_jobs: int


class OpenAIModelItem(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "qwen3-asr"


class OpenAIModelListResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIModelItem]


def to_openai_verbose_json(result: Dict[str, Any], include_words: bool = True) -> Dict[str, Any]:
    """Convert internal transcription result into OpenAI Whisper verbose_json format."""
    payload: Dict[str, Any] = {
        "task": "transcribe",
        "language": result.get("language", ""),
        "duration": result.get("duration_sec", 0.0),
        "text": result.get("text", ""),
        "segments": [
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in result.get("segments", [])
        ],
    }
    if include_words and result.get("words") is not None:
        payload["words"] = [
            {"word": w["text"], "start": w["start"], "end": w["end"]}
            for w in result["words"]
        ]
    return payload
