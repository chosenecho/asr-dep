"""Dual-GPU chunk-batch dispatcher and async long-audio job coordinator."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from app.alignment import (
    WordTimestamp,
    build_segments_from_chunk,
    merge_languages,
)
from app.audio import ChunkSpec, compute_chunks, decode_to_wav_16k
from app.config import Settings
from app.store import TaskStore
from app.worker import gpu_worker_main

logger = logging.getLogger("asr.dispatcher")


class DualGPUDispatcher:
    """
    Manages N isolated GPU worker processes (one per RTX 4090) and dispatches
    chunk mini-batches across all GPUs for maximum throughput on long recordings.
    """

    def __init__(self, settings: Settings, store: TaskStore) -> None:
        self.settings = settings
        self.store = store
        self._ctx = mp.get_context("spawn")
        self._work_queue: Any = None
        self._result_queue: Any = None
        self._ready_queue: Any = None
        self._workers: Dict[str, Any] = {}
        self._pending: Dict[str, asyncio.Future[Dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._job_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._started = False

    async def start(self) -> None:
        """Spawn per-GPU worker processes and wait until all models are loaded."""
        if self._started:
            return

        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._work_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._ready_queue = self._ctx.Queue()

        self.settings.temp_audio_dir.mkdir(parents=True, exist_ok=True)
        interrupted = self.store.mark_interrupted_tasks_failed()
        if interrupted:
            logger.warning("Marked %d interrupted tasks as failed after restart.", interrupted)

        gpu_ids = self.settings.gpu_id_list
        logger.info(
            "Starting %d worker(s) for GPUs=%s (backend=%s)...",
            len(gpu_ids),
            gpu_ids,
            self.settings.asr_backend,
        )

        for gpu_id in gpu_ids:
            worker_cfg = {
                "gpu_id": gpu_id,
                "asr_backend": self.settings.asr_backend,
                "asr_model_path": self.settings.asr_model_path,
                "aligner_model_path": self.settings.aligner_model_path,
                "gpu_memory_utilization": self.settings.gpu_memory_utilization,
                "max_inference_batch_size": self.settings.max_inference_batch_size,
                "max_new_tokens": self.settings.max_new_tokens,
                "use_flash_attention": self.settings.use_flash_attention,
                "log_level": self.settings.log_level,
            }
            proc = self._ctx.Process(
                target=gpu_worker_main,
                args=(worker_cfg, self._work_queue, self._result_queue, self._ready_queue),
                name=f"qwen3-asr-gpu-{gpu_id}",
                daemon=False,
            )
            proc.start()
            self._workers[gpu_id] = proc

        # Wait for all workers to finish model loading
        ready_count = 0
        deadline = time.monotonic() + self.settings.worker_startup_timeout_sec
        while ready_count < len(gpu_ids):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await self.stop()
                raise RuntimeError(
                    f"Timed out waiting for GPU workers to start after "
                    f"{self.settings.worker_startup_timeout_sec}s."
                )
            try:
                msg = await asyncio.to_thread(
                    self._ready_queue.get, True, min(1.0, remaining)
                )
            except queue.Empty:
                # Check if any worker died unexpectedly
                for gid, proc in self._workers.items():
                    if not proc.is_alive():
                        await self.stop()
                        raise RuntimeError(
                            f"Worker for GPU {gid} exited prematurely (exitcode={proc.exitcode})."
                        )
                continue

            if msg.get("status") != "ready":
                await self.stop()
                raise RuntimeError(
                    f"Worker for GPU {msg.get('gpu_id')} failed to initialize: {msg.get('error')}"
                )
            ready_count += 1
            logger.info(
                "Worker ready on GPU %s (pid=%s) [%d/%d].",
                msg["gpu_id"],
                msg.get("pid"),
                ready_count,
                len(gpu_ids),
            )

        self._reader_thread = threading.Thread(
            target=self._result_reader_loop,
            name="asr-result-reader",
            daemon=True,
        )
        self._reader_thread.start()
        self._started = True

    async def stop(self) -> None:
        """Gracefully terminate all background jobs and GPU workers."""
        self._stop_event.set()

        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        if self._work_queue is not None:
            for _ in self._workers:
                try:
                    self._work_queue.put_nowait(None)
                except Exception:
                    pass

        for gid, proc in self._workers.items():
            await asyncio.to_thread(proc.join, 10.0)
            if proc.is_alive():
                logger.warning("Terminating unresponsive worker on GPU %s.", gid)
                proc.terminate()
                await asyncio.to_thread(proc.join, 5.0)
        self._workers.clear()

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=3.0)
            self._reader_thread = None

        with self._pending_lock:
            for fut in self._pending.values():
                if not fut.done() and self._loop is not None:
                    self._loop.call_soon_threadsafe(
                        fut.set_exception, RuntimeError("Dispatcher shut down.")
                    )
            self._pending.clear()

        self._started = False

    def _result_reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                res = self._result_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            except Exception:
                break

            batch_id = res.get("batch_id")
            with self._pending_lock:
                fut = self._pending.pop(batch_id, None)
            if fut is not None and not fut.done() and self._loop is not None:
                self._loop.call_soon_threadsafe(fut.set_result, res)

    def health_info(self) -> Dict[str, Any]:
        workers = []
        alive_count = 0
        for gid, proc in self._workers.items():
            alive = proc.is_alive()
            if alive:
                alive_count += 1
            workers.append({"gpu_id": gid, "pid": proc.pid, "alive": alive})
        with self._pending_lock:
            inflight_batches = len(self._pending)
        return {
            "status": "ok" if (self._started and alive_count == len(self._workers) and alive_count > 0) else "degraded",
            "backend": self.settings.asr_backend,
            "model": self.settings.served_model_name,
            "aligner_enabled": bool((self.settings.aligner_model_path or "").strip()),
            "workers": workers,
            "inflight_batches": inflight_batches,
            "active_background_jobs": len(self._background_tasks),
        }

    async def _submit_chunk_batch(
        self,
        wav_path: Path,
        chunks: List[ChunkSpec],
        language: Optional[str],
        context: str,
        return_timestamps: bool,
    ) -> Dict[str, Any]:
        if not self._started:
            raise RuntimeError("Dispatcher is not running.")

        batch_id = uuid.uuid4().hex
        assert self._loop is not None
        fut: asyncio.Future[Dict[str, Any]] = self._loop.create_future()
        with self._pending_lock:
            self._pending[batch_id] = fut

        payload = {
            "batch_id": batch_id,
            "wav_path": str(wav_path),
            "chunks": [asdict(c) for c in chunks],
            "language": language,
            "context": context,
            "return_timestamps": return_timestamps,
        }
        self._work_queue.put(payload)

        res = await fut
        if not res.get("ok"):
            raise RuntimeError(
                f"GPU worker {res.get('gpu_id')} failed: {res.get('error', 'unknown error')}"
            )
        return res

    async def transcribe_wav_file(
        self,
        wav_path: Path,
        duration_sec: float,
        *,
        language: Optional[str] = None,
        context: str = "",
        return_timestamps: bool = True,
        include_words: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Split a 16kHz WAV into low-energy chunks, dispatch across GPUs, and assemble results."""
        t0 = time.perf_counter()
        chunks = await asyncio.to_thread(
            compute_chunks,
            wav_path,
            self.settings.chunk_target_sec,
            self.settings.chunk_search_expand_sec,
            100.0,
            self.settings.chunk_strategy,
            self.settings.vad_threshold,
            self.settings.vad_min_silence_ms,
        )
        total_chunks = len(chunks)
        if total_chunks == 0:
            return {
                "model": self.settings.served_model_name,
                "language": "",
                "duration_sec": duration_sec,
                "processing_time_sec": round(time.perf_counter() - t0, 3),
                "text": "",
                "segments": [],
                "words": [] if include_words else None,
            }

        if on_progress:
            on_progress(0, total_chunks)

        # Group chunks into mini-batches so BOTH RTX 4090 GPUs share long audio work in parallel.
        step = self.settings.chunks_per_dispatch
        batch_tasks = [
            asyncio.create_task(
                self._submit_chunk_batch(
                    wav_path=wav_path,
                    chunks=chunks[i : i + step],
                    language=language,
                    context=context,
                    return_timestamps=return_timestamps,
                )
            )
            for i in range(0, total_chunks, step)
        ]

        collected: List[Dict[str, Any]] = []
        completed_chunks = 0
        try:
            for finished in asyncio.as_completed(batch_tasks):
                batch_res = await finished
                chunk_list = batch_res["chunks"]
                collected.extend(chunk_list)
                completed_chunks += len(chunk_list)
                if on_progress:
                    on_progress(completed_chunks, total_chunks)
        except Exception:
            for t in batch_tasks:
                t.cancel()
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            raise

        collected.sort(key=lambda x: int(x["chunk_index"]))

        all_langs: List[str] = []
        all_texts: List[str] = []
        all_segments: List[Dict[str, Any]] = []
        all_words: List[Dict[str, Any]] = []
        next_seg_id = 0

        for item in collected:
            lang = str(item.get("language") or "")
            txt = str(item.get("text") or "").strip()
            all_langs.append(lang)
            if txt:
                all_texts.append(txt)

            raw_words = item.get("words") or []
            word_objs = [
                WordTimestamp(
                    text=str(w["text"]),
                    start=float(w["start"]),
                    end=float(w["end"]),
                )
                for w in raw_words
            ]
            if include_words:
                all_words.extend(w.to_dict() for w in word_objs)

            chunk_segments = build_segments_from_chunk(
                chunk_text=txt,
                words=word_objs if return_timestamps else [],
                chunk_start=float(item["start_sec"]),
                chunk_end=float(item["end_sec"]),
                start_segment_id=next_seg_id,
            )
            for seg in chunk_segments:
                all_segments.append(seg.to_dict(include_words=include_words))
            next_seg_id += len(chunk_segments)

        merged_lang = merge_languages(all_langs)
        joiner = " " if merged_lang.lower().startswith("english") else ""
        full_text = joiner.join(all_texts).strip()
        elapsed = round(time.perf_counter() - t0, 3)

        result: Dict[str, Any] = {
            "model": self.settings.served_model_name,
            "language": merged_lang,
            "duration_sec": round(duration_sec, 3),
            "processing_time_sec": elapsed,
            "rtf": round(elapsed / duration_sec, 4) if duration_sec > 0 else 0.0,
            "text": full_text,
            "segments": all_segments,
        }
        if include_words:
            result["words"] = all_words
        return result

    def enqueue_async_job(
        self,
        *,
        task_id: str,
        raw_audio_path: Path,
        language: Optional[str],
        context: str,
        return_timestamps: bool,
        include_words: bool,
        webhook_url: Optional[str],
    ) -> None:
        """Schedule background decoding + dual-GPU transcription for a long recording."""
        task = asyncio.create_task(
            self._run_async_job(
                task_id=task_id,
                raw_audio_path=raw_audio_path,
                language=language,
                context=context,
                return_timestamps=return_timestamps,
                include_words=include_words,
                webhook_url=webhook_url,
            ),
            name=f"asr-job-{task_id}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _run_async_job(
        self,
        *,
        task_id: str,
        raw_audio_path: Path,
        language: Optional[str],
        context: str,
        return_timestamps: bool,
        include_words: bool,
        webhook_url: Optional[str],
    ) -> None:
        wav_path = self.settings.temp_audio_dir / f"{task_id}_16k.wav"
        try:
            async with self._job_semaphore:
                self.store.update_status(task_id, status="decoding", progress=0.02)
                duration_sec = await decode_to_wav_16k(raw_audio_path, wav_path)
                self.store.update_status(
                    task_id,
                    status="processing",
                    duration_sec=duration_sec,
                    progress=0.05,
                )

                def _on_progress(done: int, total: int) -> None:
                    frac = 0.05 + 0.94 * (done / float(total)) if total > 0 else 0.99
                    self.store.update_status(
                        task_id,
                        status="processing",
                        total_chunks=total,
                        completed_chunks=done,
                        progress=frac,
                    )

                result = await self.transcribe_wav_file(
                    wav_path,
                    duration_sec,
                    language=language,
                    context=context,
                    return_timestamps=return_timestamps,
                    include_words=include_words,
                    on_progress=_on_progress,
                )
                self.store.complete_task(task_id, result)
        except Exception as exc:
            logger.exception("Async job %s failed", task_id)
            self.store.fail_task(task_id, str(exc))
        finally:
            raw_audio_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)
            self.store.cleanup_expired_tasks(self.settings.task_retention_hours)

        if webhook_url:
            await self._deliver_webhook(task_id, webhook_url)

    async def _deliver_webhook(self, task_id: str, webhook_url: str) -> None:
        task_data = self.store.get_task(task_id)
        if not task_data:
            return
        try:
            async with httpx.AsyncClient(timeout=self.settings.webhook_timeout_sec) as client:
                resp = await client.post(webhook_url, json=task_data)
                resp.raise_for_status()
            logger.info("Webhook delivered for task %s -> %s", task_id, webhook_url)
        except Exception as exc:
            logger.warning("Webhook delivery failed for task %s (%s): %s", task_id, webhook_url, exc)
