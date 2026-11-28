"""Own job lifetimes, admission tickets and prepared audio until readers drain."""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from fastapi import HTTPException, UploadFile

from app.admission import AdmissionController
from app.audio_pipeline import AudioPreparation, PreparedAudio
from app.config import Settings
from app.dispatcher import DualGPUDispatcher
from app.store import TaskStore

logger = logging.getLogger("asr.jobs")
_TERMINAL = {"completed", "failed", "cancelled"}


async def _drain_operation(awaitable: Any) -> tuple[Any, bool]:
    """Await owned work to completion without cancelling its underlying side effect."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            if task.done() and task.cancelled():
                raise
            cancelled = True


@dataclass
class _Job:
    task_id: str
    audio: PreparedAudio
    deadline: float
    deadline_at: float
    persistent: bool
    language: Optional[str]
    context: str
    return_timestamps: bool
    include_words: bool
    webhook_url: Optional[str] = None
    task: Optional[asyncio.Task[Any]] = None
    started: bool = False
    cancel_requested: bool = False


class JobCoordinator:
    def __init__(self, settings: Settings, store: TaskStore,
                 dispatcher: DualGPUDispatcher, admission: AdmissionController,
                 audio: AudioPreparation) -> None:
        self.settings = settings
        self.store = store
        self.dispatcher = dispatcher
        self.admission = admission
        self.audio = audio
        self._jobs: dict[str, _Job] = {}
        self._submissions: set[asyncio.Task[Any]] = set()
        self._retained: dict[str, _Job] = {}
        self._maintenance: Optional[asyncio.Task[Any]] = None

    @property
    def active_background_jobs(self) -> int:
        return sum(job.persistent for job in self._jobs.values())

    async def start(self) -> None:
        interrupted = await self.store.mark_interrupted_tasks_failed()
        if interrupted:
            logger.warning("Marked %d interrupted jobs failed", interrupted)
        # Only generated UUID directories belong to this coordinator. Do not touch
        # uploads or arbitrary files in a user-selected temporary directory.
        await asyncio.to_thread(self._cleanup_orphans)
        await self.store.cleanup_expired_tasks(self.settings.task_retention_hours)
        self._maintenance = asyncio.create_task(self._maintain(), name="asr-maintenance")

    def _cleanup_orphans(self) -> None:
        import shutil
        root = self.settings.temp_audio_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        for entry in root.iterdir():
            if re.fullmatch(r"job_[0-9a-f]{32}", entry.name) and entry.is_dir() and not entry.is_symlink():
                # Resolve before removal; symlinked parents cannot redirect this target.
                if entry.resolve().parent == root:
                    shutil.rmtree(entry)

    async def _maintain(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await self.store.cleanup_expired_tasks(self.settings.task_retention_hours)
            except Exception:
                logger.exception("Task retention cleanup failed")

    async def stop(self) -> None:
        self.admission.stop()
        # Staging and DB submission already own files/tickets even before they
        # appear in the runnable-job registry.
        preparing = list(self._submissions)
        for task in preparing:
            task.cancel()
        if preparing:
            await asyncio.gather(*preparing, return_exceptions=True)
        if self._maintenance:
            self._maintenance.cancel()
            await asyncio.gather(self._maintenance, return_exceptions=True)
        tasks = [job.task for job in list(self._jobs.values()) if job.task is not None]
        for job in list(self._jobs.values()):
            job.cancel_requested = True
            # Cancelling an asyncio Task before its first instruction skips its
            # finally block. A not-yet-started job observes the flag on entry.
            if job.task and job.started:
                job.task.cancel()
        # Cleanup includes dispatcher cancellation, which reaps timed-out readers.
        # Never abandon it and delete their audio while GPU processes still read.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _stage(self, upload: UploadFile, persistent: bool, **options: Any) -> _Job:
        task_id = uuid.uuid4().hex
        current = asyncio.current_task()
        assert current is not None
        self._submissions.add(current)
        ticket = None
        now = time.monotonic()
        try:
            ticket = self.admission.reserve()
            async with asyncio.timeout(self.settings.job_timeout_sec):
                prepared = await self.audio.stage(upload, task_id, ticket)
        except BaseException as exc:
            self._submissions.discard(current)
            if ticket is not None:
                ticket.release()
            await upload.close()
            if isinstance(exc, TimeoutError):
                raise HTTPException(504, detail={"code": "JOB_TIMEOUT", "message": "Upload staging deadline exceeded."}) from exc
            raise
        return _Job(task_id, prepared, now + self.settings.job_timeout_sec,
                    time.time() + max(0, self.settings.job_timeout_sec - (time.monotonic() - now)),
                    persistent, **options)

    async def submit(self, upload: UploadFile, *, language: Optional[str] = None,
                     context: str = "", return_timestamps: bool = True,
                     include_words: bool = True, webhook_url: Optional[str] = None) -> dict[str, Any]:
        if not self.settings.enable_async_tasks:
            await upload.close()
            raise HTTPException(503, detail={"code": "ASYNC_TASKS_DISABLED", "message":
                "This deployment provides synchronous transcription only."})
        job = await self._stage(upload, True, language=language, context=context,
                                return_timestamps=return_timestamps, include_words=include_words,
                                webhook_url=webhook_url)
        try:
            # Cancelling an executor Future does not roll back its SQL transaction.
            # Finish INSERT first, then either register its runner or cancel the row.
            record, cancelled = await _drain_operation(self.store.create_task(
                task_id=job.task_id, filename=upload.filename or "audio",
                language_requested=language, context=context,
                return_timestamps=return_timestamps, include_words=include_words,
                webhook_url=webhook_url, deadline_at=job.deadline_at))
            if cancelled:
                await _drain_operation(self.store.request_cancel(job.task_id))
                await _drain_operation(self.store.cancel_task(job.task_id))
                raise asyncio.CancelledError()
            self._jobs[job.task_id] = job
            job.task = asyncio.create_task(self._run(job), name=f"asr-job-{job.task_id}")
            job.task.add_done_callback(self._background_done)
            return record
        except BaseException:
            await _drain_operation(self._release(job))
            raise
        finally:
            self._submissions.discard(asyncio.current_task())

    @staticmethod
    def _background_done(task: asyncio.Task[Any]) -> None:
        if not task.cancelled() and task.exception() is not None:
            error = task.exception()
            logger.error("Background job ended with cleanup failure", exc_info=(type(error), error, error.__traceback__))

    async def transcribe(self, upload: UploadFile, *, language: Optional[str] = None,
                         context: str = "", return_timestamps: bool = True,
                         include_words: bool = True) -> dict[str, Any]:
        job = await self._stage(upload, False, language=language, context=context,
                                return_timestamps=return_timestamps, include_words=include_words)
        self._jobs[job.task_id] = job
        job.task = asyncio.current_task()
        self._submissions.discard(job.task)
        return await self._run(job)

    async def cancel(self, task_id: str) -> dict[str, Any]:
        record = await self.store.get_task(task_id)
        if record is None:
            raise HTTPException(404, "Task not found.")
        if record["status"] in ("cancelling", "cancelled"):
            return record
        if record["status"] in _TERMINAL:
            raise HTTPException(409, "Task has already reached a terminal state.")
        if not await self.store.request_cancel(task_id):
            latest = await self.store.get_task(task_id)
            if latest is not None and latest["status"] in ("cancelling", "cancelled"):
                return latest
            raise HTTPException(409, "Task has already reached a terminal state.")
        job = self._jobs.get(task_id)
        if job:
            job.cancel_requested = True
            if job.task and job.started:
                job.task.cancel()
        # Return the accepted state even if the background task completes cleanup
        # before this response is serialized.
        return (await self.store.get_task(task_id)) or record

    async def _release(self, job: _Job) -> None:
        try:
            await self.audio.release(job.audio)
        except Exception:
            # Failed deletion still consumes storage; do not advertise it as free.
            self._retained[job.task_id] = job
            self.admission.stop()
            logger.critical("Audio deletion failed for %s; retaining quota", job.task_id, exc_info=True)
            raise
        job.audio.ticket.release()
        self._jobs.pop(job.task_id, None)

    async def _run(self, job: _Job) -> dict[str, Any]:
        job.started = True
        outcome = "failed"
        error: Optional[BaseException] = None
        result: dict[str, Any] = {}
        try:
            async with asyncio.timeout_at(job.deadline):
                if job.cancel_requested:
                    raise asyncio.CancelledError()
                await job.audio.ticket.activate()
                if job.persistent:
                    await self.store.update_status(job.task_id, status="decoding", progress=0.02)
                duration = await self.audio.decode(job.audio, max_duration_sec=(
                    self.settings.max_async_audio_sec if job.persistent else self.settings.max_sync_audio_sec))
                if job.persistent:
                    await self.store.update_status(job.task_id, status="processing", duration_sec=duration,
                                                   progress=0.05)
                last_progress = -1.0

                async def progress(done: int, total: int) -> None:
                    nonlocal last_progress
                    now = time.monotonic()
                    if job.persistent and (done in (0, total) or now - last_progress >= 0.25):
                        await self.store.update_status(job.task_id, status="processing", total_chunks=total,
                            completed_chunks=done, progress=0.05 + 0.94 * done / max(1, total))
                        last_progress = now

                result = await self.dispatcher.transcribe_wav_file(
                    job.audio.wav_path, duration, job_id=job.task_id,
                    language=job.language, context=job.context,
                    return_timestamps=job.return_timestamps, include_words=job.include_words,
                    on_progress=progress)
                outcome = "completed"
        except asyncio.CancelledError as exc:
            outcome, error = "cancelled", exc
        except Exception as exc:
            error = exc
            logger.warning("Job %s failed: %s", job.task_id, exc)
        finally:
            # Shield the full drain and file release, not just the unlink, so a
            # repeated cancellation cannot interrupt resource ownership transfer.
            cleanup = asyncio.create_task(self._finish(job, outcome, result, error))
            while True:
                try:
                    await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError as exc:
                    error = error or exc
                    outcome = "cancelled"
        if not job.persistent and error is not None:
            if isinstance(error, asyncio.CancelledError):
                raise error
            if isinstance(error, HTTPException):
                raise error
            if isinstance(error, TimeoutError):
                raise HTTPException(504, detail={"code": "JOB_TIMEOUT", "message": "Job deadline exceeded."}) from error
            if isinstance(error, ValueError):
                raise HTTPException(400, f"{error} Use POST /v1/asr/tasks for longer audio.") from error
            raise HTTPException(503, detail={"code": getattr(error, "code", "INFERENCE_FAILED"),
                                           "message": str(error)}) from error
        return result

    async def _finish(self, job: _Job, outcome: str, result: dict[str, Any],
                      error: Optional[BaseException]) -> None:
        try:
            await self.dispatcher.cancel_job(job.task_id)
        except Exception as exc:
            # A failed reap is not proof that WAV readers have stopped. Keep the
            # ticket and files, stop new admission and let full shutdown reclaim.
            self._retained[job.task_id] = job
            self.admission.stop()
            logger.critical("Reader reclamation failed for %s; retaining audio", job.task_id, exc_info=True)
            if job.persistent:
                await self.store.fail_task(job.task_id, str(exc), error_code="CLEANUP_FAILED")
            raise
        try:
            if job.persistent:
                if outcome == "completed":
                    if not await self.store.complete_task(job.task_id, result):
                        await self.store.cancel_task(job.task_id)
                elif outcome == "cancelled":
                    await self.store.request_cancel(job.task_id)
                    await self.store.cancel_task(job.task_id)
                else:
                    code = "JOB_TIMEOUT" if isinstance(error, TimeoutError) else getattr(error, "code", "TASK_FAILED")
                    await self.store.fail_task(job.task_id, str(error), error_code=code)
        finally:
            await self._release(job)
        if job.webhook_url:
            await self._deliver_webhook(job.task_id, job.webhook_url)

    @property
    def unreclaimed_jobs(self) -> int:
        return len(self._retained)

    async def release_after_dispatcher_stop(self) -> None:
        """Called only after dispatcher.stop has confirmed every reader exited."""
        for job in list(self._retained.values()):
            await self._release(job)
        self._retained.clear()

    async def _deliver_webhook(self, task_id: str, url: str) -> None:
        record = await self.store.get_task(task_id)
        if record is None:
            return
        try:
            async with httpx.AsyncClient(timeout=self.settings.webhook_timeout_sec) as client:
                response = await client.post(url, json=record)
                response.raise_for_status()
        except Exception:
            logger.warning("Best-effort webhook delivery failed for %s", task_id, exc_info=True)
