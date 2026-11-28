"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator, model_validator
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
        default=16,
        ge=1,
        le=128,
        description="Max chunks per vLLM ASR inference batch inside each GPU worker (supports cross-job coalescing)",
    )
    aligner_batch_size: int = Field(
        default=6,
        ge=1,
        le=64,
        description="Max chunks per ForcedAligner GPU forward pass (decoupled from ASR batch size to prevent OOM)",
    )
    max_new_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Max generated tokens per audio chunk (2048 comfortably covers 120-150s speech)",
    )
    vllm_max_model_len: int | None = Field(
        default=None,
        ge=2048,
        le=65536,
        description="Optional vLLM max_model_len override (e.g. 16384 on 24GB L4/4090 to reduce minimum KV cache)",
    )
    vllm_enforce_eager: bool = Field(
        default=False,
        description="Disable vLLM torch.compile and CUDA Graph capture (faster cold start without GPU snapshots)",
    )
    vllm_cudagraph_mode: str = Field(
        default="",
        description="Optional vLLM CompilationConfig.cudagraph_mode override (e.g. FULL_DECODE_ONLY, PIECEWISE, FULL, NONE)",
    )
    vllm_cudagraph_capture_sizes: str = Field(
        default="",
        description="Optional comma-separated CUDA Graph batch sizes (e.g. '1,2,4,8,16') to trim capture time and VRAM/snapshot size",
    )
    vllm_compilation_mode: int | None = Field(
        default=None,
        ge=0,
        le=3,
        description="Optional vLLM CompilationMode override (0=NONE, 1=STOCK_TORCH_COMPILE, 2=DYNAMO_TRACE_ONCE, 3=VLLM_COMPILE)",
    )
    warmup_on_startup: bool = Field(
        default=False,
        description="Run a lightweight ASR + ForcedAligner + Silero VAD warmup pass on startup before reporting ready",
    )
    chunks_per_dispatch: int = Field(
        default=6,
        ge=1,
        le=32,
        description="Number of chunks per fair-scheduled work unit dispatched from each job",
    )
    batch_coalesce_wait_ms: int = Field(
        default=15,
        ge=0,
        le=500,
        description="Max milliseconds a GPU worker waits to coalesce queued chunks across jobs into a super-batch",
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
    chunk_strategy: Literal["vad", "energy"] = Field(
        default="vad",
        description="Boundary selection strategy for long audio: 'vad' (Silero VAD guided, lossless) or 'energy'",
    )
    chunk_target_sec: float = Field(
        default=120.0,
        ge=10.0,
        le=165.0,
        description="Target chunk length in seconds (must stay below ForcedAligner 180s limit)",
    )
    chunk_search_expand_sec: float = Field(
        default=15.0,
        ge=0.5,
        le=25.0,
        description="Half-window in seconds around target cut point to search for pause/silence boundary",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Silero VAD speech probability threshold",
    )
    vad_min_silence_ms: int = Field(
        default=150,
        ge=50,
        le=2000,
        description="Minimum non-speech pause duration (ms) preferred for Silero VAD boundary snapping",
    )
    vad_max_workers: int = Field(
        default=6,
        ge=1,
        le=32,
        description="Max CPU threads for parallel lock-free Silero VAD boundary search",
    )
    max_concurrent_decodes: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Max concurrent ffmpeg audio decoding subprocesses",
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

    # End-to-end resource admission and execution deadlines (single API process).
    max_queued_jobs: int = Field(default=32, ge=0)
    max_concurrent_uploads: int = Field(default=4, ge=1)
    max_temp_storage_mb: int = Field(default=8192, ge=1)
    max_decoded_audio_mb: int = Field(default=1024, ge=1)
    max_async_audio_sec: float = Field(default=28800.0, gt=0)
    decode_timeout_sec: float = Field(default=600.0, gt=0)
    batch_timeout_sec: float = Field(default=300.0, gt=0)
    batch_queue_timeout_sec: float = Field(default=600.0, gt=0)
    job_timeout_sec: float = Field(default=7200.0, gt=0)
    worker_restart_limit: int = Field(default=3, ge=0)
    shutdown_grace_sec: float = Field(default=30.0, gt=0)
    enable_async_tasks: bool = Field(default=True)

    @model_validator(mode="after")
    def _validate_resource_relationships(self) -> "Settings":
        if self.chunks_per_dispatch > self.max_inference_batch_size:
            raise ValueError("CHUNKS_PER_DISPATCH must not exceed MAX_INFERENCE_BATCH_SIZE")
        if self.chunk_target_sec + 2 * self.chunk_search_expand_sec > 170:
            raise ValueError("CHUNK_TARGET_SEC + 2 * CHUNK_SEARCH_EXPAND_SEC must be <= 170")
        if self.job_timeout_sec < max(self.batch_timeout_sec, self.batch_queue_timeout_sec):
            raise ValueError("JOB_TIMEOUT_SEC must cover both batch execution and queue deadlines")
        if len(self.gpu_id_list) != len(set(self.gpu_id_list)):
            raise ValueError("GPU_IDS must not contain duplicate devices")
        return self

    # Persistence & temporary files
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for SQLite database and temporary audio staging",
    )
    temp_dir: str = Field(
        default="",
        description="Optional override directory for temporary audio WAV files (e.g. /dev/shm/qwen3_asr_tmp)",
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
        if self.temp_dir.strip():
            return Path(self.temp_dir.strip())
        return self.data_dir / "tmp_audio"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
