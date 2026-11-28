"""FastAPI application for Dual-4090 Qwen3-ASR-1.7B + ForcedAligner long-audio API."""

from __future__ import annotations

import ipaddress
import logging
import socket
import asyncio
from contextlib import asynccontextmanager
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
from app.admission import AdmissionController
from app.audio_pipeline import AudioPreparation
from app.jobs import JobCoordinator
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


class UploadLimitMiddleware:
    """Bound upload concurrency and streamed bytes before multipart spooling."""

    _paths = {"/v1/asr/tasks", "/v1/asr/transcribe", "/v1/audio/transcriptions"}

    def __init__(self, app: Any, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        self.active_uploads = 0

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self._paths:
            await self.app(scope, receive, send)
            return
        if scope["path"] == "/v1/asr/tasks" and not self.settings.enable_async_tasks:
            await JSONResponse({"detail": {"code": "ASYNC_TASKS_DISABLED", "message":
                "This deployment provides synchronous transcription only."}}, status_code=503)(scope, receive, send)
            return
        limit = self.settings.max_upload_bytes + 64 * 1024
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)(scope, receive, send)
            return
        if declared > limit:
            await JSONResponse({"detail": "Uploaded request exceeds size limit."}, status_code=413)(scope, receive, send)
            return
        if self.active_uploads >= self.settings.max_concurrent_uploads:
            await JSONResponse({"detail": {"code": "UPLOAD_CAPACITY", "message": "Too many uploads."}},
                               status_code=503, headers={"Retry-After": "5"})(scope, receive, send)
            return
        self.active_uploads += 1
        upload_counted = True
        received = 0
        exceeded = False
        cancellation: Optional[asyncio.CancelledError] = None
        replacement = b'{"detail":"Uploaded request exceeds size limit."}'

        def release_upload() -> None:
            nonlocal upload_counted
            if upload_counted:
                self.active_uploads -= 1
                upload_counted = False

        scope["asr.release_upload"] = release_upload

        async def guarded_receive() -> Any:
            nonlocal received, exceeded, cancellation
            from starlette.formparsers import MultiPartException
            try:
                message = await receive()
            except asyncio.CancelledError as exc:
                # Let Starlette close partially spooled files before propagating
                # cancellation; its parser catches MultiPartException only.
                cancellation = exc
                raise MultiPartException("Upload cancelled.") from exc
            if message["type"] == "http.disconnect":
                # Multipart parser closes its partially spooled files on this exception.
                raise MultiPartException("Upload disconnected.")
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise MultiPartException("Uploaded request exceeds size limit.")
            return message

        async def guarded_send(message: Any) -> None:
            if cancellation is not None:
                return
            if exceeded:
                if message["type"] == "http.response.start":
                    message = {"type": "http.response.start", "status": 413, "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(replacement)).encode())]}
                elif message["type"] == "http.response.body":
                    if message.get("more_body", False):
                        return
                    message = {"type": "http.response.body", "body": replacement}
            await send(message)
        try:
            await self.app(scope, guarded_receive, guarded_send)
        finally:
            release_upload()
            if cancellation is not None:
                raise cancellation


async def _release_upload_slot(request: Request) -> None:
    # FastAPI resolves dependencies after multipart parsing, before endpoint work.
    callback = request.scope.get("asr.release_upload")
    if callback:
        callback()


def create_app(settings_override: Optional[Settings] = None) -> FastAPI:
    settings = settings_override or get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = TaskStore(settings.db_path)
        dispatcher = DualGPUDispatcher(settings=settings)
        admission = AdmissionController(settings)
        audio = AudioPreparation(settings, admission)
        jobs = JobCoordinator(settings, store, dispatcher, admission, audio)
        try:
            await store.start()
            await dispatcher.start()
            await jobs.start()
            app.state.settings = settings
            app.state.store = store
            app.state.dispatcher = dispatcher
            app.state.jobs = jobs
            app.state.admission = admission
            app.state.audio = audio
            yield
        finally:
            try:
                await jobs.stop()
            finally:
                try:
                    await dispatcher.stop()
                    await jobs.release_after_dispatcher_stop()
                finally:
                    await store.close()

    app = FastAPI(
        dependencies=[Depends(_release_upload_slot)],
        title="Qwen3-ASR-1.7B API Service",
        description=(
            "Production FastAPI service for Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B "
            "optimized for Dual RTX 4090 long-audio transcription with timestamps."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(UploadLimitMiddleware, settings=settings)

    if settings_override is not None:
        app.dependency_overrides[get_settings] = lambda: settings_override

    def _get_dispatcher(request: Request) -> DualGPUDispatcher:
        return request.app.state.dispatcher

    def _get_store(request: Request) -> TaskStore:
        return request.app.state.store

    def _get_jobs(request: Request) -> JobCoordinator:
        return request.app.state.jobs

    # ------------------------------------------------------------------
    # Health & Models
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponseSchema, tags=["System"])
    async def health_check(
        request: Request,
        response: Response,
        dispatcher: DualGPUDispatcher = Depends(_get_dispatcher),
    ) -> Dict[str, Any]:
        info = dispatcher.health_info()
        info.update(request.app.state.admission.stats())
        info["decoding_jobs"] = request.app.state.audio.decoding_jobs
        info["active_background_jobs"] = request.app.state.jobs.active_background_jobs
        info["unreclaimed_jobs"] = request.app.state.jobs.unreclaimed_jobs
        if info["unreclaimed_jobs"]:
            info["status"] = "degraded"
            info["readiness"] = False
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
        jobs: JobCoordinator = Depends(_get_jobs),
    ) -> Dict[str, Any]:
        clean_webhook = await asyncio.to_thread(_validate_webhook_url, webhook_url)
        return await jobs.submit(file, language=language, context=context,
                                 return_timestamps=return_timestamps,
                                 include_words=include_words, webhook_url=clean_webhook)

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
        return await store.list_tasks(limit=limit, status=task_status)

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
        task = await store.get_task(task_id)
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
        task = await store.get_task(task_id)
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
        task = await store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
        if task["status"] in ("queued", "decoding", "processing", "cancelling"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete a task while it is still running.",
            )
        await store.delete_task(task_id)
        return {"deleted": True, "task_id": task_id}

    @app.post(
        "/v1/asr/tasks/{task_id}/cancel", response_model=TaskStatusSchema,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_api_key)], tags=["Long-Audio Async Tasks"],
    )
    async def cancel_task(task_id: str, jobs: JobCoordinator = Depends(_get_jobs)) -> Dict[str, Any]:
        return await jobs.cancel(task_id)

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
        jobs: JobCoordinator = Depends(_get_jobs),
    ) -> Response:
        result = await jobs.transcribe(file, language=language, context=context,
                                       return_timestamps=return_timestamps, include_words=include_words)

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
        jobs: JobCoordinator = Depends(_get_jobs),
    ) -> Response:
        del model  # Accepted for OpenAI SDK compatibility; routed to loaded Qwen3-ASR-1.7B.
        granularities = set(timestamp_granularities or [])
        need_ts = response_format in ("srt", "vtt", "verbose_json") or bool(granularities)
        need_words = ("word" in granularities) or (response_format == "verbose_json" and not granularities)

        result = await jobs.transcribe(file, language=language, context=prompt,
                                       return_timestamps=need_ts, include_words=need_words)

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
