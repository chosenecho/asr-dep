"""Job lifecycle tests use explicit events rather than timing inference."""
from __future__ import annotations

import asyncio
import io
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from app.admission import AdmissionController
from app.audio_pipeline import AudioPreparation
from app.config import Settings
from app.jobs import JobCoordinator
from app.store import TaskStore


class ControlledDispatcher:
    def __init__(self):
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.reader_done = asyncio.Event()
        self.reader_done.set()
        self.paths: list[Path] = []
        self.cancelled: list[str] = []
        self.failure = None

    async def transcribe_wav_file(self, path, duration, **kwargs):
        self.paths.append(path)
        self.started.set()
        await self.finish.wait()
        if self.failure:
            raise self.failure
        await kwargs["on_progress"](1, 1)
        return {"model": "mock", "language": "Chinese", "duration_sec": duration,
                "processing_time_sec": 0.1, "text": "hello", "segments": [], "words": []}

    async def cancel_job(self, job_id):
        self.cancelled.append(job_id)
        await self.reader_done.wait()
        assert all(path.exists() for path in self.paths), "WAV removed before readers drained"


async def make_coordinator(tmp_path, **overrides):
    values = dict(asr_backend="mock", data_dir=tmp_path, gpu_ids="0", chunk_strategy="energy",
                  max_concurrent_jobs=1, max_queued_jobs=1, max_temp_storage_mb=8, _env_file=None)
    values.update(overrides)
    settings = Settings(**values)
    store = TaskStore(settings.db_path)
    await store.start()
    admission = AdmissionController(settings)
    audio = AudioPreparation(settings, admission)

    async def decode(prepared, max_duration_sec):
        prepared.wav_path.write_bytes(b"owned audio")
        prepared.duration_sec = 1.0
        return 1.0
    audio.decode = decode
    dispatcher = ControlledDispatcher()
    jobs = JobCoordinator(settings, store, dispatcher, admission, audio)
    await jobs.start()
    return jobs, dispatcher, admission, store


def upload():
    return UploadFile(io.BytesIO(b"source"), filename="sample.wav")


async def assert_clean(jobs, admission):
    assert not jobs._jobs
    assert admission.stats() == {"active_jobs": 0, "waiting_jobs": 0, "reserved_temp_bytes": 0}
    assert not list(jobs.settings.temp_audio_dir.glob("job_*"))


def test_cancel_queued_and_prestart_task_cleans_audio_and_ticket(tmp_path):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path)
        blocker = admission.reserve()
        await blocker.activate()
        record = await jobs.submit(upload())
        task = jobs._jobs[record["task_id"]].task
        try:
            accepted = await jobs.cancel(record["task_id"])
            assert accepted["status"] in {"cancelling", "cancelled"}
            await asyncio.wait_for(task, 1)
            final = await store.get_task(record["task_id"])
            assert final["status"] == "cancelled"
            assert final["cancel_requested_at"] is not None
            again = await jobs.cancel(record["task_id"])
            assert again["status"] == "cancelled"
            assert not dispatcher.started.is_set()
            blocker.release()
            await assert_clean(jobs, admission)
        finally:
            blocker.release()
            await jobs.stop()
            await store.close()
    asyncio.run(scenario())


def test_cancellation_retains_audio_until_readers_confirm_and_tolerates_repeated_cancel(tmp_path):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path)
        dispatcher.reader_done.clear()
        record = await jobs.submit(upload())
        task = jobs._jobs[record["task_id"]].task
        try:
            await asyncio.wait_for(dispatcher.started.wait(), 1)
            await jobs.cancel(record["task_id"])
            # Deliberately deliver a second cancellation during shielded cleanup.
            await asyncio.sleep(0)
            task.cancel()
            assert dispatcher.paths[0].exists()
            assert admission.active_jobs == 1
            dispatcher.reader_done.set()
            await asyncio.wait_for(task, 1)
            assert (await store.get_task(record["task_id"]))["status"] == "cancelled"
            await assert_clean(jobs, admission)
        finally:
            dispatcher.reader_done.set()
            await jobs.stop()
            await store.close()
    asyncio.run(scenario())


def test_complete_and_terminal_cancel_conflict(tmp_path):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path)
        record = await jobs.submit(upload())
        task = jobs._jobs[record["task_id"]].task
        try:
            dispatcher.finish.set()
            await asyncio.wait_for(task, 1)
            final = await store.get_task(record["task_id"])
            assert final["status"] == "completed"
            assert final["progress"] == 1
            assert final["result"]["text"] == "hello"
            with pytest.raises(HTTPException) as caught:
                await jobs.cancel(record["task_id"])
            assert caught.value.status_code == 409
            await assert_clean(jobs, admission)
        finally:
            await jobs.stop()
            await store.close()
    asyncio.run(scenario())


def test_job_timeout_fails_and_reclaims_all_resources(tmp_path):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(
            tmp_path, job_timeout_sec=0.1, batch_timeout_sec=0.05, batch_queue_timeout_sec=0.05)
        record = await jobs.submit(upload())
        task = jobs._jobs[record["task_id"]].task
        try:
            await asyncio.wait_for(task, 2)
            final = await store.get_task(record["task_id"])
            assert final["status"] == "failed"
            assert final["error_code"] == "JOB_TIMEOUT"
            await assert_clean(jobs, admission)
        finally:
            await jobs.stop()
            await store.close()
    asyncio.run(scenario())


def test_shutdown_cancels_prestart_job_and_preserves_unmanaged_files(tmp_path):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path)
        unmanaged = jobs.settings.temp_audio_dir / "keep.wav"
        unmanaged.write_bytes(b"user content")
        record = await jobs.submit(upload())
        try:
            await jobs.stop()
            assert (await store.get_task(record["task_id"]))["status"] == "cancelled"
            await assert_clean(jobs, admission)
            assert unmanaged.read_bytes() == b"user content"
            with pytest.raises(HTTPException):
                admission.reserve()
        finally:
            await jobs.stop()
            await store.close()
    asyncio.run(scenario())


def test_sync_and_async_jobs_share_admission_capacity(tmp_path):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path, max_queued_jobs=0)
        sync = asyncio.create_task(jobs.transcribe(upload()))
        try:
            await asyncio.wait_for(dispatcher.started.wait(), 1)
            with pytest.raises(HTTPException) as caught:
                await jobs.submit(upload())
            assert caught.value.status_code == 503
            dispatcher.finish.set()
            assert (await sync)["text"] == "hello"
            await assert_clean(jobs, admission)
        finally:
            dispatcher.finish.set()
            await jobs.stop()
            await asyncio.gather(sync, return_exceptions=True)
            await store.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("persistent", [True, False], ids=["async", "sync"])
def test_shutdown_cancels_staging_before_job_registration(tmp_path, monkeypatch, persistent):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path)
        source = upload()
        original_read = source.read
        staging = asyncio.Event()
        allow_eof = asyncio.Event()
        reads = 0

        async def controlled_read(size=-1):
            nonlocal reads
            reads += 1
            if reads == 1:
                return await original_read(size)
            # The first chunk has now reserved storage and been written. This
            # pause is inside AudioPreparation.stage's own cancellation cleanup.
            staging.set()
            await allow_eof.wait()
            return b""

        monkeypatch.setattr(source, "read", controlled_read)
        operation = jobs.submit if persistent else jobs.transcribe
        submission = asyncio.create_task(operation(source))
        try:
            await asyncio.wait_for(staging.wait(), 1)
            assert admission.waiting_jobs == 1
            assert admission.reserved_temp_bytes == len(b"source")
            assert not jobs._jobs
            assert list(jobs.settings.temp_audio_dir.glob("job_*"))
            await asyncio.wait_for(jobs.stop(), 2)
            assert submission.done()
            assert submission.cancelled()
            assert source.file.closed
            assert not jobs._submissions
            assert not dispatcher.started.is_set()
            assert await store.list_tasks() == []
            await assert_clean(jobs, admission)
        finally:
            allow_eof.set()
            dispatcher.finish.set()
            submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            await jobs.stop()
            await store.close()
    asyncio.run(scenario())


def test_cancel_submission_during_sql_commit_finishes_row_and_reclaims_audio(tmp_path, monkeypatch):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path)
        inserted = asyncio.Event()
        allow_commit = threading.Event()
        loop = asyncio.get_running_loop()
        original_row_to_dict = store._row_to_dict

        def pause_insert_result(row, include_result=True):
            # create_task converts its row after INSERT but before leaving its
            # transaction. Cancellation of its Future must not orphan the commit.
            loop.call_soon_threadsafe(inserted.set)
            if not allow_commit.wait(timeout=5):
                raise AssertionError("test did not release the SQLite commit barrier")
            return original_row_to_dict(row, include_result=include_result)

        monkeypatch.setattr(store, "_row_to_dict", pause_insert_result)
        submission = asyncio.create_task(jobs.submit(upload()))
        try:
            await asyncio.wait_for(inserted.wait(), 2)
            assert list(jobs.settings.temp_audio_dir.glob("job_*"))
            submission.cancel()
            cancellation_delivered = asyncio.Event()
            loop.call_soon(cancellation_delivered.set)
            await asyncio.wait_for(cancellation_delivered.wait(), 1)
            assert not submission.done(), "submission abandoned its in-progress SQL transaction"
            assert admission.waiting_jobs == 1
            assert admission.reserved_temp_bytes == len(b"source")
            allow_commit.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(submission, 2)
            rows = await store.list_tasks()
            assert len(rows) == 1
            assert rows[0]["status"] == "cancelled"
            assert rows[0]["cancel_requested_at"] is not None
            assert rows[0]["completed_at"] is not None
            assert not dispatcher.started.is_set()
            assert not jobs._submissions
            await assert_clean(jobs, admission)
        finally:
            # Unblock the DB thread before awaiting coordinator/store shutdown,
            # including assertion-failure paths; no worker can outlive this test.
            allow_commit.set()
            dispatcher.finish.set()
            submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            await jobs.stop()
            await store.close()
    asyncio.run(scenario())


def test_failed_reader_reclamation_retains_files_until_dispatcher_stopped(tmp_path, monkeypatch):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path)
        dispatcher.reader_done.clear()
        record = await jobs.submit(upload())
        task_id = record["task_id"]
        runner = jobs._jobs[task_id].task
        original_cancel = dispatcher.cancel_job

        async def failed_reclamation(job_id):
            raise OSError("worker reclamation failed")

        try:
            await asyncio.wait_for(dispatcher.started.wait(), 1)
            monkeypatch.setattr(dispatcher, "cancel_job", failed_reclamation)
            await jobs.cancel(task_id)
            with pytest.raises(OSError, match="worker reclamation failed"):
                await asyncio.wait_for(runner, 2)
            assert not dispatcher.reader_done.is_set()
            assert dispatcher.paths[0].exists()
            assert dispatcher.paths[0].with_name("audio.raw").exists()
            assert admission.active_jobs == 1
            assert admission.reserved_temp_bytes == len(b"source")
            assert jobs.unreclaimed_jobs == 1
            assert task_id in jobs._jobs
            assert admission.stopped
            final = await store.get_task(task_id)
            assert final["status"] == "failed"
            assert final["error_code"] == "CLEANUP_FAILED"
            await jobs.stop()
            assert dispatcher.paths[0].exists(), "jobs.stop alone does not prove readers exited"
            assert admission.active_jobs == 1
            # Simulate the full dispatcher stop having now confirmed reader exit.
            dispatcher.reader_done.set()
            await jobs.release_after_dispatcher_stop()
            assert jobs.unreclaimed_jobs == 0
            await assert_clean(jobs, admission)
            await jobs.release_after_dispatcher_stop()  # Idempotent final release.
        finally:
            dispatcher.reader_done.set()
            dispatcher.finish.set()
            monkeypatch.setattr(dispatcher, "cancel_job", original_cancel)
            await jobs.stop()
            await asyncio.gather(runner, return_exceptions=True)
            await jobs.release_after_dispatcher_stop()
            await store.close()
    asyncio.run(scenario())


def test_failed_audio_deletion_keeps_storage_reservation(tmp_path, monkeypatch):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path)
        record = await jobs.submit(upload())
        task = jobs._jobs[record["task_id"]].task
        release = jobs.audio.release

        async def denied_delete(prepared):
            raise PermissionError("audio still locked")

        try:
            await asyncio.wait_for(dispatcher.started.wait(), 1)
            monkeypatch.setattr(jobs.audio, "release", denied_delete)
            await jobs.cancel(record["task_id"])
            with pytest.raises(PermissionError, match="audio still locked"):
                await asyncio.wait_for(task, 1)
            assert admission.active_jobs == 1
            assert admission.reserved_temp_bytes == len(b"source")
            assert dispatcher.paths[0].exists()
            assert jobs.unreclaimed_jobs == 1
            assert admission.stopped
            monkeypatch.setattr(jobs.audio, "release", release)
            await jobs.release_after_dispatcher_stop()
            await assert_clean(jobs, admission)
        finally:
            monkeypatch.setattr(jobs.audio, "release", release)
            await jobs.stop()
            await jobs.release_after_dispatcher_stop()
            await store.close()
    asyncio.run(scenario())


def test_concurrent_duplicate_cancel_is_idempotent(tmp_path, monkeypatch):
    async def scenario():
        jobs, dispatcher, admission, store = await make_coordinator(tmp_path)
        dispatcher.reader_done.clear()
        record = await jobs.submit(upload())
        task_id = record["task_id"]
        runner = jobs._jobs[task_id].task
        both_read_active = asyncio.Event()
        original_get = store.get_task
        reads = 0
        requests = []

        async def concurrent_initial_read(requested_id):
            nonlocal reads
            row = await original_get(requested_id)
            if requested_id == task_id and reads < 2:
                reads += 1
                assert row["status"] == "processing"
                if reads == 2:
                    both_read_active.set()
                await both_read_active.wait()
            return row

        try:
            await asyncio.wait_for(dispatcher.started.wait(), 1)
            monkeypatch.setattr(store, "get_task", concurrent_initial_read)
            requests = [asyncio.create_task(jobs.cancel(task_id)) for _ in range(2)]
            accepted = await asyncio.wait_for(asyncio.gather(*requests), 2)
            assert all(row["status"] == "cancelling" for row in accepted)
            assert not runner.done()
            assert dispatcher.paths[0].exists()
            dispatcher.reader_done.set()
            await asyncio.wait_for(runner, 2)
            assert (await store.get_task(task_id))["status"] == "cancelled"
            await assert_clean(jobs, admission)
        finally:
            both_read_active.set()
            dispatcher.reader_done.set()
            dispatcher.finish.set()
            monkeypatch.setattr(store, "get_task", original_get)
            await asyncio.gather(*requests, return_exceptions=True)
            await jobs.stop()
            await asyncio.gather(runner, return_exceptions=True)
            await store.close()
    asyncio.run(scenario())
