"""Resource accounting and ffmpeg cancellation tests using asyncio.run."""
import asyncio
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException, UploadFile

from app.admission import AdmissionController
from app.audio import SAMPLE_RATE, decode_to_wav_16k
from app.audio_pipeline import AudioPreparation


def settings(tmp_path, **overrides):
    values = dict(max_concurrent_jobs=2, max_queued_jobs=1, max_temp_storage_mb=2,
                  max_concurrent_decodes=1, max_decoded_audio_mb=1,
                  max_upload_bytes=1024 * 1024, temp_audio_dir=tmp_path,
                  decode_timeout_sec=2, asr_backend="mock")
    values.update(overrides)
    return SimpleNamespace(**values)


def wav_upload(seconds=0.1):
    stream = BytesIO()
    sf.write(stream, np.zeros(int(seconds * SAMPLE_RATE)), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    stream.seek(0)
    return UploadFile(stream, filename="input.wav")


def test_stage_decode_reconcile_and_idempotent_release(tmp_path, monkeypatch):
    monkeypatch.setattr("app.audio.shutil.which", lambda name: None)
    async def run():
        config = settings(tmp_path)
        admission = AdmissionController(config)
        pipeline = AudioPreparation(config, admission)
        ticket = admission.reserve()
        prepared = await pipeline.stage(wav_upload(), "../../not-a-path", ticket)
        assert prepared.raw_path.parent.name.startswith("job_")
        assert admission.reserved_temp_bytes == prepared.source_bytes
        await ticket.activate()
        assert await pipeline.decode(prepared, 1) == 0.1
        assert admission.reserved_temp_bytes == prepared.actual_bytes
        assert pipeline.decoding_jobs == 0
        await pipeline.release(prepared)
        await pipeline.release(prepared)
        ticket.release()
        assert admission.reserved_temp_bytes == 0
        assert list(tmp_path.iterdir()) == []
    asyncio.run(run())


def test_failed_upload_returns_reserved_bytes_and_removes_directory(tmp_path):
    async def run():
        config = settings(tmp_path, max_temp_storage_mb=0)
        admission = AdmissionController(config)
        pipeline = AudioPreparation(config, admission)
        ticket = admission.reserve()
        upload = wav_upload()
        with pytest.raises(HTTPException) as error:
            await pipeline.stage(upload, "request", ticket)
        assert error.value.status_code == 503
        assert upload.file.closed
        assert admission.reserved_temp_bytes == 0
        assert list(tmp_path.iterdir()) == []
        ticket.release()
    asyncio.run(run())


def test_duration_limit_does_not_accept_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr("app.audio.shutil.which", lambda name: None)
    async def run():
        config = settings(tmp_path)
        admission = AdmissionController(config)
        pipeline = AudioPreparation(config, admission)
        ticket = admission.reserve()
        prepared = await pipeline.stage(wav_upload(0.5), "request", ticket)
        with pytest.raises(ValueError, match="duration"):
            await pipeline.decode(prepared, 0.1)
        assert not prepared.wav_path.exists()
        assert admission.reserved_temp_bytes == prepared.source_bytes
        await pipeline.release(prepared)
        ticket.release()
    asyncio.run(run())


def test_cancelled_upload_cleans_partial_file_and_budget(tmp_path):
    async def run():
        second_read = asyncio.Event()

        class Upload:
            closed = False
            reads = 0

            async def read(self, size):
                self.reads += 1
                if self.reads == 1:
                    return b"first chunk"
                second_read.set()
                await asyncio.Event().wait()

            async def close(self):
                self.closed = True

        config = settings(tmp_path)
        admission = AdmissionController(config)
        pipeline = AudioPreparation(config, admission)
        ticket = admission.reserve()
        upload = Upload()
        task = asyncio.create_task(pipeline.stage(upload, "request", ticket))
        await second_read.wait()
        assert admission.reserved_temp_bytes == len(b"first chunk")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert upload.closed
        assert admission.reserved_temp_bytes == 0
        assert list(tmp_path.iterdir()) == []
        ticket.release()
    asyncio.run(run())


def test_production_requires_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("app.audio.shutil.which", lambda name: None)
    with pytest.raises(ValueError, match="ffmpeg is required"):
        asyncio.run(decode_to_wav_16k(tmp_path / "input", tmp_path / "output.wav"))


def test_shared_decode_semaphore_and_cancelled_waiter_release(tmp_path, monkeypatch):
    async def run():
        entered = asyncio.Event()
        finish = asyncio.Event()
        calls = []

        async def decode(source, destination, **kwargs):
            calls.append(source)
            entered.set()
            await finish.wait()
            sf.write(destination, np.zeros(1600), SAMPLE_RATE, subtype="PCM_16")
            return 0.1

        monkeypatch.setattr("app.audio_pipeline.decode_to_wav_16k", decode)
        config = settings(tmp_path)
        admission = AdmissionController(config)
        pipeline = AudioPreparation(config, admission)
        tickets = [admission.reserve(), admission.reserve()]
        audios = [await pipeline.stage(wav_upload(), str(i), ticket)
                  for i, ticket in enumerate(tickets)]
        first = asyncio.create_task(pipeline.decode(audios[0], 1))
        await entered.wait()
        second = asyncio.create_task(pipeline.decode(audios[1], 1))
        await asyncio.sleep(0)
        assert len(calls) == pipeline.decoding_jobs == 1
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        finish.set()
        await first
        for audio, ticket in zip(audios, tickets):
            await pipeline.release(audio)
            ticket.release()
        assert admission.reserved_temp_bytes == 0
        assert not list(tmp_path.iterdir())
    asyncio.run(run())


def test_real_ffmpeg_rejects_duration_and_byte_truncation(tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    source = tmp_path / "source.wav"
    output = tmp_path / "out.wav"
    sf.write(source, np.zeros(SAMPLE_RATE * 2), SAMPLE_RATE, subtype="PCM_16")
    with pytest.raises(ValueError, match="duration"):
        asyncio.run(decode_to_wav_16k(source, output, max_duration_sec=0.1))
    assert not output.exists()
    with pytest.raises(ValueError, match="byte limit"):
        asyncio.run(decode_to_wav_16k(source, output, max_output_bytes=10000))
    assert not output.exists()
    assert asyncio.run(decode_to_wav_16k(source, output)) == 2.0


@pytest.mark.parametrize("first_failure", ["cancel", "timeout"])
@pytest.mark.parametrize("repeat_during", ["process_wait", "stderr_drain"])
def test_repeated_cancel_waits_for_process_and_pipe_before_unlink(
        tmp_path, monkeypatch, first_failure, repeat_during):
    async def run():
        started = asyncio.Event()
        terminated = asyncio.Event()
        permit_exit = asyncio.Event()
        process_reaped = asyncio.Event()
        permit_pipe_eof = asyncio.Event()
        output = tmp_path / "audio.wav"

        class Stderr:
            async def read(self, count):
                await permit_pipe_eof.wait()
                return b""

        class Process:
            returncode = None
            stderr = Stderr()

            def terminate(self):
                terminated.set()

            def kill(self):
                permit_exit.set()

            async def wait(self):
                await permit_exit.wait()
                self.returncode = -15
                process_reaped.set()
                return self.returncode

        process = Process()

        async def spawn(*args, **kwargs):
            output.write_bytes(b"partial")
            started.set()
            return process

        monkeypatch.setattr("app.audio.shutil.which", lambda name: "ffmpeg")
        monkeypatch.setattr("app.audio.asyncio.create_subprocess_exec", spawn)
        task = asyncio.create_task(decode_to_wav_16k(
            tmp_path / "source", output,
            timeout_sec=0.01 if first_failure == "timeout" else 5))
        await started.wait()
        if first_failure == "cancel":
            task.cancel("initial cancellation")
        await asyncio.wait_for(terminated.wait(), 1)
        if repeat_during == "stderr_drain":
            permit_exit.set()
            await process_reaped.wait()
        task.cancel("repeated cancellation during cleanup")
        await asyncio.sleep(0)
        assert not task.done()
        assert output.exists()
        permit_exit.set()
        await process_reaped.wait()
        await asyncio.sleep(0)
        assert not task.done()
        assert output.exists(), "File must survive until stderr draining also finishes."
        permit_pipe_eof.set()
        expected = asyncio.CancelledError if first_failure == "cancel" else TimeoutError
        with pytest.raises(expected) as error:
            await task
        if first_failure == "cancel":
            assert error.value.args == ("initial cancellation",)
        assert process.returncode is not None
        assert not output.exists()
    asyncio.run(run())


@pytest.mark.parametrize("cancel", [True, False])
def test_ffmpeg_cancel_and_timeout_reap_before_cleanup(tmp_path, monkeypatch, cancel):
    async def run():
        started = asyncio.Event()
        stopped = asyncio.Event()
        output = tmp_path / "audio.wav"

        class Stderr:
            async def read(self, count):
                await stopped.wait()
                return b""

        class Process:
            returncode = None
            stderr = Stderr()
            terminated = False
            waited = False

            def terminate(self):
                self.terminated = True
                self.returncode = -15
                stopped.set()

            def kill(self):
                self.returncode = -9
                stopped.set()

            async def wait(self):
                await stopped.wait()
                self.waited = True
                return self.returncode

        process = Process()

        async def spawn(*args, **kwargs):
            output.write_bytes(b"partial")
            started.set()
            return process

        monkeypatch.setattr("app.audio.shutil.which", lambda name: "ffmpeg")
        monkeypatch.setattr("app.audio.asyncio.create_subprocess_exec", spawn)
        task = asyncio.create_task(decode_to_wav_16k(
            tmp_path / "source", output, timeout_sec=0.01 if not cancel else 5))
        await started.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        assert process.terminated and process.waited
        assert not output.exists()
    asyncio.run(run())
