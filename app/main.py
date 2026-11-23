"""FastAPI application for Dual-4090 Qwen3-ASR-1.7B + ForcedAligner long-audio API."""

from __future__ import annotations

import ipaddress
import logging
import socket
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, PlainTextResponse

from app.alignment import format_srt, format_vtt
from app.audio import decode_to_wav_16k, save_upload_file
from app.auth import verify_api_key
from app.config import Settings, get_settings
from app.dispatcher import DualGPUDispatcher
from app.schemas import (
    HealthResponseSchema,
    OpenAIModelItem,
    OpenAIModelListResponse,
    TaskStatusSchema,
    TranscriptionResultSchema,
    to_openai_verbose_json,
)
from app.store import TaskStore


def _validate_webhook_url(webhook_url: Optional[str]) -> Optional[str]:
    """Validate optional webhook URL and block non-HTTP(S) or private/loopback targets (SSRF guard)."""
    if not webhook_url or not webhook_url.strip():
        return None
    url = webhook_url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="webhook_url must be a valid http(s) URL.",
        )

    host = parsed.hostname
    if host.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="webhook_url cannot target loopback addresses.",
        )
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="webhook_url cannot target private or internal network addresses.",
                )
    except HTTPException:
        raise
    except OSError:
        # Allow DNS resolution failures at submission time only if not obviously private IP
        pass
    return url


def _safe_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".bin"
    suffix = Path(filename).suffix.lower()
    if suffix and len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".bin"


def create_app(settings_override: Optional[Settings] = None) -> FastAPI:
    settings = settings_override or get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = TaskStore(settings.db_path)
        dispatcher = DualGPUDispatcher(settings=settings, store=store)
        await dispatcher.start()
        app.state.settings = settings
        app.state.store = store
        app.state.dispatcher = dispatcher
        try:
            yield
        finally:
            await dispatcher.stop()

    app = FastAPI(
        title="Qwen3-ASR-1.7B API Service",
        description=(
            "Production FastAPI service for Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B "
            "optimized for Dual RTX 4090 long-audio transcription with timestamps."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    if settings_override is not None:
        app.dependency_overrides[get_settings] = lambda: settings_override

    def _get_dispatcher(request: Request) -> DualGPUDispatcher:
        return request.app.state.dispatcher

    def _get_store(request: Request) -> TaskStore:
        return request.app.state.store

    # ------------------------------------------------------------------
    # Health & Models
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponseSchema, tags=["System"])
    async def health_check(
        response: Response,
        dispatcher: DualGPUDispatcher = Depends(_get_dispatcher),
    ) -> Dict[str, Any]:
        info = dispatcher.health_info()
        if info["status"] != "ok":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return info

    @app.get(
        "/v1/models",
        response_model=OpenAIModelListResponse,
        dependencies=[Depends(verify_api_key)],
        tags=["OpenAI Compatible"],
    )
    async def list_models() -> OpenAIModelListResponse:
        return OpenAIModelListResponse(
            data=[OpenAIModelItem(id=settings.served_model_name)]
        )

    # ------------------------------------------------------------------
    # Async Long-Audio Task API (Recommended for long recordings)
    # ------------------------------------------------------------------

    @app.post(
        "/v1/asr/tasks",
        response_model=TaskStatusSchema,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_api_key)],
        tags=["Long-Audio Async Tasks"],
    )
    async def create_transcription_task(
        file: UploadFile = File(..., description="Audio or video file (mp3/wav/m4a/mp4/flac/ogg/...)"),
        language: Optional[str] = Form(default=None, description="Language name or ISO code (e.g. Chinese, English, zh, en); omit for auto-detect"),
        context: str = Form(default="", description="Optional domain context / hotwords prompt"),
        return_timestamps: bool = Form(default=True, description="Enable ForcedAligner timestamps"),
        include_words: bool = Form(default=True, description="Include word/char-level timestamps in response"),
        webhook_url: Optional[str] = Form(default=None, description="Optional HTTP(S) callback URL on completion"),
        store: TaskStore = Depends(_get_store),
        dispatcher: DualGPUDispatcher = Depends(_get_dispatcher),
    ) -> Dict[str, Any]:
        clean_webhook = _validate_webhook_url(webhook_url)
        task_id = uuid.uuid4().hex
        raw_path = settings.temp_audio_dir / f"{task_id}_raw{_safe_suffix(file.filename)}"
        await save_upload_file(file, raw_path, settings.max_upload_bytes)

        task_record = store.create_task(
            task_id=task_id,
            filename=file.filename or "audio",
            language_requested=language,
            context=context,
            return_timestamps=return_timestamps,
            include_words=include_words,
            webhook_url=clean_webhook,
        )
        dispatcher.enqueue_async_job(
            task_id=task_id,
            raw_audio_path=raw_path,
            language=language,
            context=context,
            return_timestamps=return_timestamps,
            include_words=include_words,
            webhook_url=clean_webhook,
        )
        return task_record

    @app.get(
        "/v1/asr/tasks",
        response_model=List[TaskStatusSchema],
        dependencies=[Depends(verify_api_key)],
        tags=["Long-Audio Async Tasks"],
    )
    async def list_tasks(
        limit: int = Query(default=50, ge=1, le=200),
        task_status: Optional[str] = Query(default=None, alias="status"),
        store: TaskStore = Depends(_get_store),
    ) -> List[Dict[str, Any]]:
        return store.list_tasks(limit=limit, status=task_status)

    @app.get(
        "/v1/asr/tasks/{task_id}",
        response_model=TaskStatusSchema,
        dependencies=[Depends(verify_api_key)],
        tags=["Long-Audio Async Tasks"],
    )
    async def get_task(
        task_id: str,
        store: TaskStore = Depends(_get_store),
    ) -> Dict[str, Any]:
        task = store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
        return task

    @app.get(
        "/v1/asr/tasks/{task_id}/subtitle",
        dependencies=[Depends(verify_api_key)],
        tags=["Long-Audio Async Tasks"],
    )
    async def get_task_subtitle(
        task_id: str,
        format: Literal["srt", "vtt"] = Query(default="srt"),
        store: TaskStore = Depends(_get_store),
    ) -> PlainTextResponse:
        task = store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
        if task["status"] != "completed" or not task.get("result"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Task is not completed yet (status={task['status']}).",
            )
        segments = task["result"].get("segments", [])
        if format == "vtt":
            return PlainTextResponse(
                content=format_vtt(segments),
                media_type="text/vtt; charset=utf-8",
            )
        return PlainTextResponse(
            content=format_srt(segments),
            media_type="application/x-subrip; charset=utf-8",
        )

    @app.delete(
        "/v1/asr/tasks/{task_id}",
        dependencies=[Depends(verify_api_key)],
        tags=["Long-Audio Async Tasks"],
    )
    async def delete_task(
        task_id: str,
        store: TaskStore = Depends(_get_store),
    ) -> Dict[str, Any]:
        task = store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
        if task["status"] in ("queued", "decoding", "processing"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete a task while it is still running.",
            )
        store.delete_task(task_id)
        return {"deleted": True, "task_id": task_id}

    # ------------------------------------------------------------------
    # Synchronous ASR API (Short / Medium Recordings)
    # ------------------------------------------------------------------

    @app.post(
        "/v1/asr/transcribe",
        response_model=TranscriptionResultSchema,
        dependencies=[Depends(verify_api_key)],
        tags=["Synchronous ASR"],
    )
    async def transcribe_sync(
        file: UploadFile = File(...),
        language: Optional[str] = Form(default=None),
        context: str = Form(default=""),
        return_timestamps: bool = Form(default=True),
        include_words: bool = Form(default=True),
        response_format: Literal["json", "text", "srt", "vtt"] = Form(default="json"),
        dispatcher: DualGPUDispatcher = Depends(_get_dispatcher),
    ) -> Response:
        req_id = uuid.uuid4().hex
        raw_path = settings.temp_audio_dir / f"sync_{req_id}_raw{_safe_suffix(file.filename)}"
        wav_path = settings.temp_audio_dir / f"sync_{req_id}_16k.wav"
        try:
            await save_upload_file(file, raw_path, settings.max_upload_bytes)
            try:
                duration_sec = await decode_to_wav_16k(raw_path, wav_path)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

            if duration_sec > settings.max_sync_audio_sec:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Audio duration ({duration_sec:.1f}s) exceeds synchronous limit "
                        f"({settings.max_sync_audio_sec:.0f}s). Use POST /v1/asr/tasks instead."
                    ),
                )

            result = await dispatcher.transcribe_wav_file(
                wav_path,
                duration_sec,
                language=language,
                context=context,
                return_timestamps=return_timestamps,
                include_words=include_words,
            )
        finally:
            raw_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

        if response_format == "text":
            return PlainTextResponse(result["text"])
        if response_format == "srt":
            return PlainTextResponse(
                format_srt(result["segments"]),
                media_type="application/x-subrip; charset=utf-8",
            )
        if response_format == "vtt":
            return PlainTextResponse(
                format_vtt(result["segments"]),
                media_type="text/vtt; charset=utf-8",
            )
        return JSONResponse(content=result)

    # ------------------------------------------------------------------
    # OpenAI-Compatible Endpoint (/v1/audio/transcriptions)
    # ------------------------------------------------------------------

    @app.post(
        "/v1/audio/transcriptions",
        dependencies=[Depends(verify_api_key)],
        tags=["OpenAI Compatible"],
    )
    async def openai_audio_transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default="qwen3-asr-1.7b"),
        language: Optional[str] = Form(default=None),
        prompt: str = Form(default=""),
        response_format: Literal["json", "text", "srt", "vtt", "verbose_json"] = Form(default="json"),
        timestamp_granularities: Optional[List[str]] = Form(default=None, alias="timestamp_granularities[]"),
        dispatcher: DualGPUDispatcher = Depends(_get_dispatcher),
    ) -> Response:
        del model  # Accepted for OpenAI SDK compatibility; routed to loaded Qwen3-ASR-1.7B.
        granularities = set(timestamp_granularities or [])
        need_ts = response_format in ("srt", "vtt", "verbose_json") or bool(granularities)
        need_words = ("word" in granularities) or (response_format == "verbose_json" and not granularities)

        req_id = uuid.uuid4().hex
        raw_path = settings.temp_audio_dir / f"oai_{req_id}_raw{_safe_suffix(file.filename)}"
        wav_path = settings.temp_audio_dir / f"oai_{req_id}_16k.wav"
        try:
            await save_upload_file(file, raw_path, settings.max_upload_bytes)
            try:
                duration_sec = await decode_to_wav_16k(raw_path, wav_path)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

            if duration_sec > settings.max_sync_audio_sec:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Audio duration ({duration_sec:.1f}s) exceeds synchronous limit "
                        f"({settings.max_sync_audio_sec:.0f}s). Use POST /v1/asr/tasks instead."
                    ),
                )

            result = await dispatcher.transcribe_wav_file(
                wav_path,
                duration_sec,
                language=language,
                context=prompt,
                return_timestamps=need_ts,
                include_words=need_words,
            )
        finally:
            raw_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

        if response_format == "text":
            return PlainTextResponse(result["text"])
        if response_format == "srt":
            return PlainTextResponse(
                format_srt(result["segments"]),
                media_type="application/x-subrip; charset=utf-8",
            )
        if response_format == "vtt":
            return PlainTextResponse(
                format_vtt(result["segments"]),
                media_type="text/vtt; charset=utf-8",
            )
        if response_format == "verbose_json":
            return JSONResponse(content=to_openai_verbose_json(result, include_words=need_words))
        return JSONResponse(content={"text": result["text"]})

    return app


app = create_app()
