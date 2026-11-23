"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration for dual-GPU Qwen3-ASR + ForcedAligner serving."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, description="Bind port")
    log_level: str = Field(default="INFO", description="Logging level")

    # Authentication (comma-separated API keys; empty disables auth for local dev)
    api_keys: str = Field(
        default="",
        description="Comma-separated allowed API keys. Leave empty to disable auth.",
    )

    # Model paths (local directory recommended in production)
    asr_model_path: str = Field(
        default="Qwen/Qwen3-ASR-1.7B",
        description="Path or HF/ModelScope ID for Qwen3-ASR-1.7B",
    )
    aligner_model_path: str = Field(
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        description="Path or HF/ModelScope ID for Qwen3-ForcedAligner-0.6B (empty to disable)",
    )
    served_model_name: str = Field(
        default="qwen3-asr-1.7b",
        description="Public model name reported in API responses",
    )

    # Inference backend & dual-GPU layout
    asr_backend: Literal["vllm", "transformers", "mock"] = Field(
        default="vllm",
        description="Inference backend: vllm (recommended), transformers, or mock (for tests)",
    )
    gpu_ids: str = Field(
        default="0,1",
        description="Comma-separated physical GPU IDs (one dedicated worker process per GPU)",
    )
    gpu_memory_utilization: float = Field(
        default=0.72,
        ge=0.2,
        le=0.95,
        description="vLLM GPU memory utilization per 4090 (leave ~5-6GB for ForcedAligner)",
    )
    max_inference_batch_size: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Max chunks per inference batch inside each GPU worker",
    )
    max_new_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Max generated tokens per audio chunk (2048 comfortably covers 120-150s speech)",
    )
    chunks_per_dispatch: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of chunks dispatched as one work unit to a GPU worker",
    )
    use_flash_attention: bool = Field(
        default=False,
        description="Enable flash_attention_2 for ForcedAligner / Transformers backend",
    )
    worker_startup_timeout_sec: int = Field(
        default=300,
        ge=10,
        description="Seconds to wait for GPU workers to load models during startup",
    )

    # Long audio chunking & upload limits
    chunk_target_sec: float = Field(
        default=120.0,
        ge=10.0,
        le=165.0,
        description="Target chunk length in seconds (must stay below ForcedAligner 180s limit)",
    )
    chunk_search_expand_sec: float = Field(
        default=5.0,
        ge=0.5,
        le=15.0,
        description="Half-window in seconds to search for low-energy split boundary",
    )
    max_upload_size_mb: int = Field(
        default=500,
        ge=1,
        le=4096,
        description="Max single uploaded audio/video file size in MB",
    )
    max_sync_audio_sec: float = Field(
        default=600.0,
        ge=30.0,
        description="Recommended max audio duration (sec) for synchronous endpoints",
    )
    max_concurrent_jobs: int = Field(
        default=16,
        ge=1,
        description="Max concurrent active transcription jobs",
    )

    # Persistence & temporary files
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for SQLite database and temporary audio staging",
    )
    task_retention_hours: int = Field(
        default=72,
        ge=1,
        description="Hours to keep completed/failed task records before cleanup",
    )
    webhook_timeout_sec: float = Field(
        default=15.0,
        ge=1.0,
        description="HTTP timeout for optional task completion webhook callbacks",
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def _to_path(cls, v: str | Path) -> Path:
        return Path(v)

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def gpu_id_list(self) -> List[str]:
        ids = [g.strip() for g in self.gpu_ids.split(",") if g.strip()]
        return ids or ["0"]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tasks.db"

    @property
    def temp_audio_dir(self) -> Path:
        return self.data_dir / "tmp_audio"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
