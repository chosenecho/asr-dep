"""Deterministic CPU-only supervision tests using spawned, controlled readers."""

from __future__ import annotations

import asyncio
import functools
import os
from pathlib import Path
import queue
import threading
from unittest.mock import MagicMock, patch

import pytest

from app.audio import ChunkSpec
from app.config import Settings
from app.dispatcher import DualGPUDispatcher, DispatcherError, _Batch
from app.worker import _prefetch_loop, _run_real_with_isolation
from app.worker_protocol import (
    BatchRequest,
    BatchResult,
    BatchSkipped,
    BatchStarted,
    PrefetchFailed,
)


def controlled_worker(events, release, cfg, work, results, ready, cancelled):
    if os.name == "posix":
        os.setsid()
    gid, gen = cfg["gpu_id"], cfg["worker_generation"]
    ready.put(dict(gpu_id=gid, worker_generation=gen, status="ready"))
    while True:
        item = work.get()
        if item is None:
            return
        if item.context != "unacquired":
            results.put(BatchStarted(item.batch_id, item.attempt, gid, gen))
        events.put(item.batch_id)
        if item.context == "exit":
            # Drain the ack feeder before exiting; ownership is deterministic.
            results.close()
            results.join_thread()
            os._exit(7)
        if item.context in ("hang", "unacquired", "cancel"):
            release.wait()
        if cancelled.get(item.job_id, False):
            results.put(BatchSkipped(item.batch_id, item.attempt, gid, gen))
        else:
            result = BatchResult(item.batch_id, item.attempt, gid, gen, chunks=[])
            results.put(result)
            results.put(result)


def settings(tmp_path, **kwargs):
    cfg = Settings(
        _env_file=None,
        asr_backend="mock",
        gpu_ids="0",
        data_dir=tmp_path,
        chunk_strategy="energy",
        worker_restart_limit=0,
    )
    for name, value in kwargs.items():
        object.__setattr__(cfg, name, value)
    return cfg


async def install(tmp_path, mode, **kwargs):
    dispatcher = DualGPUDispatcher(settings(tmp_path, **kwargs))
    events = dispatcher._ctx.Queue()
    release = dispatcher._ctx.Event()
    target = functools.partial(controlled_worker, events, release)
    with patch("app.dispatcher.gpu_worker_main", target):
        await dispatcher.start()
    future = asyncio.get_running_loop().create_future()
    request = BatchRequest(
        "batch", "job", str(tmp_path / "audio.wav"), [], context=mode
    )
    dispatcher._jobs.add("job")
    dispatcher._job_queues["job"] = __import__("collections").deque(
        [_Batch(request, future)]
    )
    dispatcher._job_ring.append("job")
    dispatcher._pump_scheduler()
    assert await asyncio.wait_for(asyncio.to_thread(events.get, True, 5), 6) == "batch"
    return dispatcher, future, release


@pytest.mark.parametrize(
    "mode,code",
    [
        ("exit", "WORKER_EXITED"),
        ("hang", "BATCH_TIMEOUT"),
        ("unacquired", "BATCH_QUEUE_TIMEOUT"),
    ],
)
def test_failed_reader_reaped_before_future(tmp_path, mode, code):
    async def run():
        d, future, release = await install(
            tmp_path, mode, batch_timeout_sec=0.1, batch_queue_timeout_sec=0.1
        )
        process = d._workers["0"]
        try:
            with pytest.raises(DispatcherError) as exc:
                await asyncio.wait_for(asyncio.shield(future), 8)
            assert exc.value.code == code
            assert not process.is_alive()
            assert process.exitcode is not None
            assert d._inflight_count == 0
            assert not d._pending
            assert d.health_info()["worker_restarts"] == 0
        finally:
            await d.stop()

    asyncio.run(run())


def test_transcription_waits_for_chunk_reader_on_repeated_cancel(tmp_path):
    async def run():
        d = DualGPUDispatcher(settings(tmp_path))
        d._started = True
        entered, release = threading.Event(), threading.Event()

        def chunks(*args):
            entered.set()
            assert release.wait(3)
            return []

        with patch("app.dispatcher.compute_chunks", chunks):
            task = asyncio.create_task(
                d.transcribe_wav_file(tmp_path / "x.wav", 1, job_id="j")
            )
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        assert not d._jobs and not d._chunk_tasks and not d._cancelled

    asyncio.run(run())


def test_spawned_mock_success_awaits_progress_and_never_touches_store(tmp_path):
    class ForbiddenStore:
        def __getattr__(self, name):
            raise AssertionError(f"Store accessed: {name}")

    async def run():
        d = DualGPUDispatcher(settings(tmp_path), ForbiddenStore())
        progress = []

        async def callback(done, total):
            await asyncio.sleep(0)
            progress.append((done, total))

        await d.start()
        try:
            with patch(
                "app.dispatcher.compute_chunks",
                return_value=[ChunkSpec(0, 0, 16000, 0, 1)],
            ):
                result = await asyncio.wait_for(
                    d.transcribe_wav_file(
                        tmp_path / "mock.wav", 1, on_progress=callback, job_id="success"
                    ),
                    5,
                )
            assert result["text"]
            assert progress == [(0, 1), (1, 1)]
            assert not d._pending and d._inflight_count == 0
            assert not d._cancelled
            assert d.health_info()["readiness"]
        finally:
            await d.stop()

    asyncio.run(run())


def test_duplicates_and_stale_generation_settle_once(tmp_path):
    async def run():
        d = DualGPUDispatcher(settings(tmp_path))
        future = asyncio.get_running_loop().create_future()
        req = BatchRequest("b", "j", "x", [], worker_generation=2)
        d._pending["b"] = _Batch(req, future, gpu_id="0")
        d._inflight_count = 1
        d._accept_message(BatchResult("b", 1, "0", 1))
        d._accept_message(BatchResult("b", 2, "0", 2))
        assert not future.done()
        result = BatchResult("b", 1, "0", 2)
        d._accept_message(result)
        d._accept_message(result)
        assert (await future)["ok"]
        assert d._inflight_count == 0

    asyncio.run(run())


def test_cancel_waits_for_process_reaping(tmp_path):
    async def run():
        d, future, release = await install(tmp_path, "cancel", shutdown_grace_sec=0.05)
        process = d._workers["0"]
        try:
            await asyncio.wait_for(d.cancel_job("job"), 8)
            assert not process.is_alive()
            assert not d._pending
            with pytest.raises(DispatcherError):
                await future
        finally:
            await d.stop()

    asyncio.run(run())


def test_superbatch_isolation_keeps_good_item_and_skips_cancelled():
    import logging

    loaded = [
        ({"batch_id": bid, "job_id": bid}, [], [])
        for bid in ("good", "bad", "cancelled")
    ]
    calls, emitted = [], {}

    def infer(**kwargs):
        items = kwargs["loaded_items"]
        calls.append([pair[0]["batch_id"] for pair in items])
        if len(items) > 1 or items[0][0]["batch_id"] == "bad":
            raise ValueError("injected model failure")
        return [(items[0][0]["batch_id"], [{"text": "ok"}])]

    def emit(item, **kwargs):
        emitted[item["batch_id"]] = kwargs

    with patch("app.worker._process_real_superbatch", infer):
        _run_real_with_isolation(
            None, loaded, logging.getLogger("test"), 1, None, {"cancelled": True}, emit
        )
    assert calls == [["good", "bad", "cancelled"], ["good"], ["bad"]]
    assert emitted["good"]["chunks"] == [{"text": "ok"}]
    assert isinstance(emitted["bad"]["error"], ValueError)
    assert emitted["cancelled"] == {}


def test_prefetch_failure_notifies_consumer():
    class BrokenQueue:
        def get(self):
            raise RuntimeError("injected queue failure")

    prefetched = queue.Queue()
    import logging

    thread = threading.Thread(
        target=_prefetch_loop,
        args=(
            BrokenQueue(),
            prefetched,
            "mock",
            1,
            1,
            0,
            None,
            logging.getLogger("test"),
        ),
    )
    thread.start()
    message = prefetched.get(timeout=2)
    thread.join(timeout=2)
    assert isinstance(message, PrefetchFailed)
    assert "injected queue failure" in message.error
    assert not thread.is_alive()


def test_failed_reap_propagates_and_retains_pending_readers(tmp_path):
    async def run():
        d = DualGPUDispatcher(settings(tmp_path, shutdown_grace_sec=0))
        process = MagicMock(pid=None)
        process.is_alive.return_value = True
        d._workers["0"] = process
        d._queues["0"] = queue.Queue()
        future = asyncio.get_running_loop().create_future()
        d._pending["b"] = _Batch(BatchRequest("b", "j", "x", []), future, gpu_id="0")
        d._inflight_count = 1
        with pytest.raises(DispatcherError, match="WORKER_REAP_FAILED"):
            await d.stop()
        assert d._workers["0"] is process
        assert "b" in d._pending and not future.done()
        future.cancel()

    asyncio.run(run())


def test_bounded_worker_restart_budget(tmp_path):
    async def run():
        d = DualGPUDispatcher(settings(tmp_path, worker_restart_limit=2))

        async def reaped(gid):
            pass

        with patch.object(d, "_reap_worker", reaped):
            for _ in range(5):
                await d._fail_worker("0", "WORKER_EXITED")
        assert d._worker_restarts["0"] == 2

    asyncio.run(run())
