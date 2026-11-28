"""Exercise ingress limits against FastAPI's real Starlette multipart parser."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, File, UploadFile
import starlette.formparsers

from app.main import UploadLimitMiddleware, _release_upload_slot

PATHS = ["/v1/asr/tasks", "/v1/asr/transcribe", "/v1/audio/transcriptions"]
BOUNDARY = b"upload-test-boundary"
PREFIX = (b"--" + BOUNDARY + b'\r\nContent-Disposition: form-data; name="file"; '
          b'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n')
SUFFIX = b"\r\n--" + BOUNDARY + b"--\r\n"


def make_app(*, max_upload_bytes=32, max_concurrent_uploads=1,
             enable_async_tasks=True, endpoint=None):
    app = FastAPI(dependencies=[Depends(_release_upload_slot)])

    async def normal(file: UploadFile = File(...)):
        return {"bytes": len(await file.read())}

    for path in PATHS:
        app.post(path)(endpoint or normal)
    config = SimpleNamespace(max_upload_bytes=max_upload_bytes,
                             max_concurrent_uploads=max_concurrent_uploads,
                             enable_async_tasks=enable_async_tasks)
    return UploadLimitMiddleware(app, config)


def scope(path=PATHS[1], *, content_length=None):
    headers = [(b"content-type", b"multipart/form-data; boundary=" + BOUNDARY)]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"", "headers": headers, "client": ("127.0.0.1", 1234),
            "server": ("test", 80), "root_path": ""}


async def invoke(app, messages, *, path=PATHS[1], content_length=None):
    sent = []
    reads = 0
    source = iter(messages)

    async def receive():
        nonlocal reads
        reads += 1
        return next(source)

    async def send(message):
        sent.append(message)

    await app(scope(path, content_length=content_length), receive, send)
    return sent, reads


def response_start(sent):
    return next(message for message in sent if message["type"] == "http.response.start")


@pytest.fixture
def spooled_files(monkeypatch):
    original = starlette.formparsers.SpooledTemporaryFile
    files = []

    def tracked(*args, **kwargs):
        # Force real disk spooling after the first bytes, so cleanup covers OS file handles.
        kwargs["max_size"] = 1
        file = original(*args, **kwargs)
        files.append(file)
        return file

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", tracked)
    return files


@pytest.mark.parametrize("path", PATHS)
def test_streamed_oversize_without_content_length_closes_partial_multipart(path, spooled_files):
    async def run():
        app = make_app()
        messages = [
            {"type": "http.request", "body": PREFIX + b"partial", "more_body": True},
            {"type": "http.request", "body": b"x" * (64 * 1024), "more_body": True},
        ]
        sent, reads = await invoke(app, messages, path=path)
        assert response_start(sent)["status"] == 413
        assert reads == 2
        assert spooled_files and all(file.closed for file in spooled_files)
        assert app.active_uploads == 0
    asyncio.run(run())


@pytest.mark.parametrize("path", PATHS)
def test_declared_oversize_never_reads_body(path, spooled_files):
    async def run():
        app = make_app()
        sent, reads = await invoke(app, [], path=path, content_length=64 * 1024 + 33)
        assert response_start(sent)["status"] == 413
        assert reads == 0
        assert not spooled_files
        assert app.active_uploads == 0
    asyncio.run(run())


def test_upload_concurrency_returns_retry_after_without_reading_second_body(spooled_files):
    async def run():
        app = make_app()
        blocked = asyncio.Event()
        finish = asyncio.Event()
        calls = 0

        async def receive():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "http.request", "body": PREFIX + b"partial", "more_body": True}
            blocked.set()
            await finish.wait()
            return {"type": "http.request", "body": SUFFIX, "more_body": False}

        async def send(message):
            pass

        first = asyncio.create_task(app(scope(), receive, send))
        await blocked.wait()
        assert app.active_uploads == 1
        sent, reads = await invoke(app, [], path=PATHS[2])
        start = response_start(sent)
        assert start["status"] == 503
        assert dict(start["headers"])[b"retry-after"]
        assert reads == 0
        finish.set()
        await first
        assert app.active_uploads == 0
        assert spooled_files and all(file.closed for file in spooled_files)
    asyncio.run(run())


def test_disconnect_closes_partial_multipart(spooled_files):
    async def run():
        app = make_app()
        sent, reads = await invoke(app, [
            {"type": "http.request", "body": PREFIX + b"partial", "more_body": True},
            {"type": "http.disconnect"},
        ])
        assert response_start(sent)["status"] == 400
        assert reads == 2
        assert spooled_files and all(file.closed for file in spooled_files)
        assert app.active_uploads == 0
    asyncio.run(run())


def test_cancelled_call_closes_partial_multipart(spooled_files):
    async def run():
        app = make_app()
        blocked = asyncio.Event()
        calls = 0

        async def receive():
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "http.request", "body": PREFIX + b"partial", "more_body": True}
            blocked.set()
            await asyncio.Event().wait()

        async def send(message):
            pass

        request = asyncio.create_task(app(scope(), receive, send))
        await blocked.wait()
        assert spooled_files and not spooled_files[0].closed
        assert spooled_files[0]._rolled
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert app.active_uploads == 0
        assert all(file.closed for file in spooled_files)
    asyncio.run(run())


def test_async_disabled_returns_503_without_reading_body(spooled_files):
    async def run():
        app = make_app(enable_async_tasks=False)
        sent, reads = await invoke(app, [], path=PATHS[0])
        assert response_start(sent)["status"] == 503
        assert b"ASYNC_TASKS_DISABLED" in b"".join(
            message.get("body", b"") for message in sent)
        assert reads == 0
        assert not spooled_files
        assert app.active_uploads == 0
    asyncio.run(run())


@pytest.mark.parametrize("path", PATHS)
def test_under_limit_parses_file_and_dependency_releases_slot_before_endpoint(path, spooled_files):
    async def run():
        app = None

        async def endpoint(file: UploadFile = File(...)):
            assert app.active_uploads == 0
            return {"bytes": len(await file.read())}

        app = make_app(endpoint=endpoint)
        sent, reads = await invoke(app, [
            {"type": "http.request", "body": PREFIX + b"audio", "more_body": True},
            {"type": "http.request", "body": SUFFIX, "more_body": False},
        ], path=path)
        assert response_start(sent)["status"] == 200
        assert b'{"bytes":5}' in [message.get("body") for message in sent]
        assert reads == 2
        assert spooled_files and all(file.closed for file in spooled_files)
        assert app.active_uploads == 0
    asyncio.run(run())
