"""Fair batch scheduling and supervised, isolated GPU readers.

A completed/cancelled transcription owns no WAV readers. Linux workers lead a
private session so reclamation includes vLLM descendants; Windows uses Process
terminate/kill (deployment must supervise any external backend descendants).
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass
import inspect
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any, Callable, Optional
import uuid

from app.alignment import WordTimestamp, build_segments_from_chunk, merge_languages
from app.audio import compute_chunks, warmup_silero_vad
from app.config import Settings
from app.worker import gpu_worker_main
from app.worker_protocol import BatchRequest, BatchStarted, BatchResult, BatchSkipped

logger = logging.getLogger("asr.dispatcher")


class DispatcherError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass
class _Batch:
    request: BatchRequest
    future: asyncio.Future
    gpu_id: str | None = None
    submitted: float | None = None
    acquired: float | None = None


async def _drain_shielded(awaitable: Any) -> Any:
    """Finish resource reclamation even if the caller is cancelled repeatedly."""
    task = asyncio.ensure_future(awaitable)
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


class DualGPUDispatcher:
    def __init__(self, settings: Settings, store: Any = None) -> None:
        self.settings = settings
        self._ctx = mp.get_context("spawn")
        self._workers: dict[str, Any] = {}
        self._queues: dict[str, Any] = {}
        self._results: dict[str, Any] = {}
        self._generations: dict[str, int] = {}
        self._worker_restarts: dict[str, int] = {}
        self._readiness: dict[str, bool] = {}
        self._worker_since: dict[str, float] = {}
        self._restart_after: dict[str, float] = {}
        self._pending: dict[str, _Batch] = {}
        self._job_queues: dict[str, deque[_Batch]] = {}
        self._job_ring: deque[str] = deque()
        self._jobs: set[str] = set()
        self._chunk_tasks: dict[str, asyncio.Task] = {}
        self._inflight_count = 0
        batches_per_super = max(
            1,
            math.ceil(
                settings.max_inference_batch_size / max(1, settings.chunks_per_dispatch)
            ),
        )
        self._max_queued_batches = max(
            4, len(settings.gpu_id_list) * batches_per_super * 2
        )
        self._manager: Any = None
        self._cancelled: Any = {}
        self._result_queue: Any = queue.Queue()
        self._ready_queue: Any = None
        self._monitor_task: asyncio.Task | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        self._stopping = False

    def _spawn_worker(self, gpu_id: str) -> None:
        generation = self._generations.get(gpu_id, 0) + 1
        self._generations[gpu_id] = generation
        names = (
            "asr_backend",
            "asr_model_path",
            "aligner_model_path",
            "gpu_memory_utilization",
            "max_inference_batch_size",
            "aligner_batch_size",
            "batch_coalesce_wait_ms",
            "max_new_tokens",
            "vllm_max_model_len",
            "vllm_enforce_eager",
            "vllm_cudagraph_mode",
            "vllm_cudagraph_capture_sizes",
            "vllm_compilation_mode",
            "warmup_on_startup",
            "use_flash_attention",
            "log_level",
        )
        cfg = {name: getattr(self.settings, name) for name in names}
        cfg.update(
            gpu_id=gpu_id,
            num_workers=len(self.settings.gpu_id_list),
            worker_generation=generation,
        )
        work_queue = self._ctx.Queue()
        result_queue = self._ctx.Queue()
        self._results[gpu_id] = result_queue
        proc = self._ctx.Process(
            target=gpu_worker_main,
            args=(cfg, work_queue, result_queue, self._ready_queue, self._cancelled),
            name=f"qwen3-asr-gpu-{gpu_id}",
            daemon=False,
        )
        proc.start()
        self._queues[gpu_id] = work_queue
        self._workers[gpu_id] = proc
        self._readiness[gpu_id] = False
        self._worker_since[gpu_id] = time.monotonic()
        result_sink = self._result_queue

        def read_results() -> None:
            # A killed multiprocessing writer may leave a partial frame/locked
            # pipe. Isolate that pipe and its daemon reader to this generation.
            while self._generations.get(gpu_id) == generation and not self._stopping:
                try:
                    result_sink.put(result_queue.get(timeout=0.1))
                except queue.Empty:
                    if not proc.is_alive():
                        return
                except (EOFError, OSError, ValueError):
                    return

        threading.Thread(
            target=read_results, name=f"asr-results-{gpu_id}-{generation}", daemon=True
        ).start()

    async def start(self) -> None:
        if self._started:
            return
        self._stopping = False
        self._manager = self._ctx.Manager()
        self._cancelled = self._manager.dict()
        self._result_queue = queue.Queue()
        self._ready_queue = self._ctx.Queue()
        try:
            for gid in self.settings.gpu_id_list:
                self._worker_restarts[gid] = 0
                self._spawn_worker(gid)
            deadline = time.monotonic() + self.settings.worker_startup_timeout_sec
            while not all(self._readiness.values()):
                self._read_ready()
                if any(not p.is_alive() for p in self._workers.values()):
                    raise DispatcherError(
                        "WORKER_EXITED", "Worker exited during startup"
                    )
                if time.monotonic() >= deadline:
                    raise DispatcherError(
                        "WORKER_STARTUP_TIMEOUT", "Model readiness deadline exceeded"
                    )
                await asyncio.sleep(0.02)
            if (
                self.settings.warmup_on_startup
                and self.settings.chunk_strategy == "vad"
                and self.settings.asr_backend != "mock"
            ):
                await asyncio.to_thread(
                    warmup_silero_vad, self.settings.vad_max_workers
                )
            self._started = True
            self._monitor_task = asyncio.create_task(
                self._monitor_loop(), name="asr-worker-monitor"
            )
        except BaseException:
            await _drain_shielded(self.stop())
            raise

    def _read_ready(self) -> None:
        while True:
            try:
                msg = self._ready_queue.get_nowait()
            except queue.Empty:
                return
            gid = str(msg["gpu_id"])
            if msg.get("worker_generation") != self._generations.get(gid):
                continue
            if msg.get("status") == "ready":
                self._readiness[gid] = True
            else:
                logger.error(
                    "Worker %s initialization failed: %s", gid, msg.get("error")
                )

    async def _reap_worker(self, gid: str, graceful: bool = False) -> None:
        proc = self._workers.get(gid)
        if proc is None:
            return
        self._readiness[gid] = False
        if graceful and proc.is_alive():
            self._queues[gid].put_nowait(None)
            await asyncio.to_thread(
                proc.join, float(getattr(self.settings, "shutdown_grace_sec", 30))
            )
        # setsid is called before loading models. A process may die before setsid;
        # its pid then names no private group, and killpg safely returns ESRCH.
        if os.name == "posix" and proc.pid:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc.is_alive():
            proc.terminate()
        await asyncio.to_thread(proc.join, 5.0)
        if proc.is_alive():
            proc.kill()
            await asyncio.to_thread(proc.join, 5.0)
        if proc.is_alive():
            raise DispatcherError(
                "WORKER_REAP_FAILED", f"Worker {gid} survived terminate/kill"
            )
        if os.name == "posix" and proc.pid and Path("/proc").is_dir():
            # SIGKILL delivery alone does not prove descendants stopped reading.
            # Linux procfs lets us exclude only zombies (which hold no files).
            def live_group_members() -> list[int]:
                members = []
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdigit():
                        continue
                    try:
                        fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    if int(fields[2]) == proc.pid and fields[0] not in ("Z", "X"):
                        members.append(int(entry.name))
                return members

            deadline = time.monotonic() + 5.0
            while await asyncio.to_thread(live_group_members):
                if time.monotonic() >= deadline:
                    raise DispatcherError(
                        "WORKER_REAP_FAILED", f"Worker {gid} descendants remain alive"
                    )
                await asyncio.sleep(0.01)
        old_queue = self._queues.pop(gid, None)
        if old_queue is not None:
            old_queue.cancel_join_thread()
            old_queue.close()
        self._workers.pop(gid, None)
        result_queue = self._results.pop(gid, None)
        if result_queue is not None:
            result_queue.cancel_join_thread()
            result_queue.close()

    def _settle(
        self,
        batch_id: str,
        result: BatchResult | BatchSkipped | None = None,
        error: Exception | None = None,
    ) -> None:
        batch = self._pending.pop(batch_id, None)
        if batch is None:
            return
        self._inflight_count -= 1
        if not batch.future.done():
            if error is not None:
                batch.future.set_exception(error)
            elif isinstance(result, BatchSkipped):
                batch.future.cancel()
            else:
                batch.future.set_result(asdict(result))

    def _accept_message(self, msg: Any) -> None:
        if not isinstance(msg, (BatchStarted, BatchResult, BatchSkipped)):
            logger.warning("Ignoring invalid worker message: %r", type(msg))
            return
        batch = self._pending.get(msg.batch_id)
        if (
            batch is None
            or batch.gpu_id != msg.gpu_id
            or batch.request.attempt != msg.attempt
            or batch.request.worker_generation != msg.worker_generation
        ):
            logger.debug("Ignoring stale/duplicate batch message %s", msg.batch_id)
            return
        if isinstance(msg, BatchStarted):
            if batch.acquired is None:
                batch.acquired = time.monotonic()
        else:
            self._settle(msg.batch_id, msg)

    async def _fail_worker(self, gid: str, code: str) -> None:
        await self._reap_worker(gid)
        # Futures are released only after all processes that could read WAV exit.
        for bid, batch in list(self._pending.items()):
            if batch.gpu_id == gid:
                self._settle(
                    bid, error=DispatcherError(code, f"GPU worker {gid} reclaimed")
                )
        used = self._worker_restarts.get(gid, 0)
        if not self._stopping and used < getattr(
            self.settings, "worker_restart_limit", 3
        ):
            self._worker_restarts[gid] = used + 1
            self._restart_after[gid] = time.monotonic() + min(0.25 * 2**used, 5.0)

    async def _monitor_loop(self) -> None:
        try:
            while not self._stopping:
                async with self._lifecycle_lock:
                    self._read_ready()
                    for _ in range(1024):
                        try:
                            self._accept_message(self._result_queue.get_nowait())
                        except queue.Empty:
                            break
                    now = time.monotonic()
                    failures: dict[str, str] = {}
                    for gid, proc in self._workers.items():
                        if not proc.is_alive():
                            failures[gid] = "WORKER_EXITED"
                        elif (
                            not self._readiness[gid]
                            and now - self._worker_since[gid]
                            > self.settings.worker_startup_timeout_sec
                        ):
                            failures[gid] = "WORKER_STARTUP_TIMEOUT"
                    for batch in self._pending.values():
                        if batch.acquired is not None:
                            if now - batch.acquired > getattr(
                                self.settings, "batch_timeout_sec", 300
                            ):
                                failures[batch.gpu_id] = "BATCH_TIMEOUT"
                        elif now - batch.submitted > getattr(
                            self.settings, "batch_queue_timeout_sec", 600
                        ):
                            failures[batch.gpu_id] = "BATCH_QUEUE_TIMEOUT"
                    for gid, code in failures.items():
                        await self._fail_worker(gid, code)
                    for gid, due in list(self._restart_after.items()):
                        if now >= due:
                            self._restart_after.pop(gid)
                            self._spawn_worker(gid)
                    if not self._workers and not self._restart_after:
                        for batches in self._job_queues.values():
                            for batch in batches:
                                if not batch.future.done():
                                    batch.future.set_exception(
                                        DispatcherError(
                                            "WORKER_UNAVAILABLE",
                                            "Worker restart budget exhausted",
                                        )
                                    )
                        self._job_queues.clear()
                        self._job_ring.clear()
                    self._pump_scheduler()
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Worker monitor failed; reclaiming readers")
            await self.stop()

    def _pump_scheduler(self) -> None:
        available = [gid for gid in self._workers if self._readiness.get(gid)]
        if not self._started or self._stopping or not available:
            return
        while self._inflight_count < self._max_queued_batches and self._job_ring:
            jid = self._job_ring.popleft()
            batches = self._job_queues.get(jid)
            if not batches:
                continue
            batch = batches.popleft()
            if batches:
                self._job_ring.append(jid)
            else:
                self._job_queues.pop(jid, None)
            if batch.future.done():
                continue
            gid = min(
                available,
                key=lambda g: sum(b.gpu_id == g for b in self._pending.values()),
            )
            batch.gpu_id = gid
            batch.request = BatchRequest(
                **{**asdict(batch.request), "worker_generation": self._generations[gid]}
            )
            batch.submitted = time.monotonic()
            self._pending[batch.request.batch_id] = batch
            self._inflight_count += 1
            self._queues[gid].put_nowait(batch.request)

    async def cancel_job(self, job_id: str) -> None:
        if (
            job_id not in self._jobs
            and job_id not in self._job_queues
            and not any(b.request.job_id == job_id for b in self._pending.values())
        ):
            return
        self._cancelled[job_id] = True
        chunk_task = self._chunk_tasks.get(job_id)
        if chunk_task is not None:
            await asyncio.gather(asyncio.shield(chunk_task), return_exceptions=True)
        for batch in self._job_queues.pop(job_id, ()):
            batch.future.cancel()
        self._job_ring = deque(j for j in self._job_ring if j != job_id)
        deadline = time.monotonic() + getattr(self.settings, "shutdown_grace_sec", 30)
        while any(b.request.job_id == job_id for b in self._pending.values()):
            if time.monotonic() >= deadline:
                async with self._lifecycle_lock:
                    gids = {
                        b.gpu_id
                        for b in self._pending.values()
                        if b.request.job_id == job_id
                    }
                    for gid in gids:
                        await self._fail_worker(gid, "JOB_CANCELLED")
                break
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        """Return only after every worker tree and local WAV reader has ended.

        Reaping errors propagate with workers/pending records retained, so callers
        must retain the corresponding files and may retry shutdown.
        """
        self._stopping = True
        monitor = self._monitor_task
        if monitor is not None and monitor is not asyncio.current_task():
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        self._monitor_task = None
        async with self._lifecycle_lock:
            for gid in list(self._workers):
                await self._reap_worker(gid, graceful=True)
            for bid in list(self._pending):
                self._settle(
                    bid,
                    error=DispatcherError("DISPATCHER_SHUTDOWN", "Dispatcher stopped"),
                )
            for batches in self._job_queues.values():
                for batch in batches:
                    if not batch.future.done():
                        batch.future.set_exception(
                            DispatcherError("DISPATCHER_SHUTDOWN", "Dispatcher stopped")
                        )
            self._job_queues.clear()
            self._job_ring.clear()
            self._restart_after.clear()
            if self._chunk_tasks:
                await asyncio.gather(
                    *self._chunk_tasks.values(), return_exceptions=True
                )
            if self._manager is not None:
                await asyncio.to_thread(self._manager.shutdown)
                self._manager = None
                self._cancelled = {}
            for ipc in (self._ready_queue,):
                if ipc is not None:
                    ipc.cancel_join_thread()
                    ipc.close()
            self._ready_queue = self._result_queue = None
            self._started = False

    def health_info(self) -> dict[str, Any]:
        workers = [
            {
                "gpu_id": gid,
                "pid": self._workers[gid].pid if gid in self._workers else None,
                "alive": gid in self._workers and self._workers[gid].is_alive(),
                "ready": self._readiness.get(gid, False),
                "generation": self._generations.get(gid, 0),
            }
            for gid in self.settings.gpu_id_list
        ]
        ready = (
            self._started
            and not self._stopping
            and all(w["alive"] and w["ready"] for w in workers)
        )
        return {
            "status": "ok" if ready else "degraded",
            "backend": self.settings.asr_backend,
            "model": self.settings.served_model_name,
            "aligner_enabled": bool((self.settings.aligner_model_path or "").strip()),
            "workers": workers,
            "inflight_batches": len(self._pending)
            + sum(map(len, self._job_queues.values())),
            "active_background_jobs": 0,
            "worker_restarts": sum(self._worker_restarts.values()),
            "readiness": bool(ready),
        }

    async def transcribe_wav_file(
        self,
        wav_path: Path,
        duration_sec: float,
        *,
        language: Optional[str] = None,
        context: str = "",
        return_timestamps: bool = True,
        include_words: bool = True,
        on_progress: Optional[Callable[[int, int], Any]] = None,
        job_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not self._started or self._stopping:
            raise DispatcherError("WORKER_UNAVAILABLE", "Dispatcher is not running")
        jid = job_id or uuid.uuid4().hex
        if jid in self._jobs:
            raise ValueError(f"Job already running: {jid}")
        self._jobs.add(jid)
        t0 = time.perf_counter()
        batch_futures: list[asyncio.Future] = []
        collected: list[dict[str, Any]] = []

        async def progress(done: int, total: int) -> None:
            if on_progress:
                value = on_progress(done, total)
                if inspect.isawaitable(value):
                    await value

        # Chunk discovery also reads the WAV; cancellation must join its thread.
        chunk_task = asyncio.create_task(
            asyncio.to_thread(
                compute_chunks,
                wav_path,
                self.settings.chunk_target_sec,
                self.settings.chunk_search_expand_sec,
                100.0,
                self.settings.chunk_strategy,
                self.settings.vad_threshold,
                self.settings.vad_min_silence_ms,
                None,
                self.settings.vad_max_workers,
            )
        )
        self._chunk_tasks[jid] = chunk_task
        try:
            chunks = await asyncio.shield(chunk_task)
            if self._stopping:
                raise DispatcherError(
                    "DISPATCHER_SHUTDOWN", "Dispatcher stopped during chunk discovery"
                )
            if self._cancelled.get(jid, False):
                raise asyncio.CancelledError()
            await progress(0, len(chunks))
            if self._stopping:
                raise DispatcherError(
                    "DISPATCHER_SHUTDOWN", "Dispatcher stopped before batch submission"
                )
            if self._cancelled.get(jid, False):
                raise asyncio.CancelledError()
            batches: deque[_Batch] = deque()
            for i in range(0, len(chunks), self.settings.chunks_per_dispatch):
                request = BatchRequest(
                    uuid.uuid4().hex,
                    jid,
                    str(wav_path),
                    [
                        asdict(c)
                        for c in chunks[i : i + self.settings.chunks_per_dispatch]
                    ],
                    language,
                    context,
                    return_timestamps,
                )
                future = asyncio.get_running_loop().create_future()
                batch_futures.append(future)
                batches.append(_Batch(request, future))
            if batches:
                self._job_queues[jid] = batches
                self._job_ring.append(jid)
                self._pump_scheduler()
            completed = 0
            for future in asyncio.as_completed(batch_futures):
                result = await future
                if not result["ok"]:
                    raise DispatcherError("INFERENCE_FAILED", str(result.get("error")))
                collected.extend(result["chunks"])
                completed += len(result["chunks"])
                await progress(completed, len(chunks))
        except BaseException:
            await _drain_shielded(asyncio.gather(chunk_task, return_exceptions=True))
            await _drain_shielded(self.cancel_job(jid))
            await _drain_shielded(
                asyncio.gather(*batch_futures, return_exceptions=True)
            )
            raise
        finally:
            self._jobs.discard(jid)
            self._chunk_tasks.pop(jid, None)
            if not any(b.request.job_id == jid for b in self._pending.values()):
                self._cancelled.pop(jid, None)
            self._job_queues.pop(jid, None)
        return self._assemble(
            collected, duration_sec, t0, return_timestamps, include_words
        )

    def _assemble(
        self,
        collected: list[dict[str, Any]],
        duration_sec: float,
        t0: float,
        return_timestamps: bool,
        include_words: bool,
    ) -> dict[str, Any]:
        collected.sort(key=lambda x: int(x["chunk_index"]))
        langs, texts, segments, words = [], [], [], []
        for item in collected:
            langs.append(str(item.get("language") or ""))
            text = str(item.get("text") or "").strip()
            if text:
                texts.append(text)
            word_objs = [
                WordTimestamp(
                    text=str(w["text"]), start=float(w["start"]), end=float(w["end"])
                )
                for w in item.get("words") or []
            ]
            if include_words:
                words.extend(w.to_dict() for w in word_objs)
            built = build_segments_from_chunk(
                chunk_text=text,
                words=word_objs if return_timestamps else [],
                chunk_start=float(item["start_sec"]),
                chunk_end=float(item["end_sec"]),
                start_segment_id=len(segments),
            )
            segments.extend(s.to_dict(include_words=include_words) for s in built)
        lang = merge_languages(langs)
        elapsed = round(time.perf_counter() - t0, 3)
        result = {
            "model": self.settings.served_model_name,
            "language": lang,
            "duration_sec": round(duration_sec, 3),
            "processing_time_sec": elapsed,
            "rtf": round(elapsed / duration_sec, 4) if duration_sec else 0.0,
            "text": (" " if lang.lower().startswith("english") else "")
            .join(texts)
            .strip(),
            "segments": segments,
        }
        if include_words:
            result["words"] = words
        return result
