"""End-to-end integration tests with 2 spawned mock GPU workers and FastAPI TestClient."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.audio import SAMPLE_RATE
from app.config import Settings
from app.main import create_app


def _make_wav_bytes(duration_sec: float) -> bytes:
    frames = int(round(duration_sec * SAMPLE_RATE))
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    wav = (0.25 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture(scope="module")
def api_client(tmp_path_factory: pytest.TempPathFactory):
    data_dir: Path = tmp_path_factory.mktemp("asr_data")
    settings = Settings(
        asr_backend="mock",
        gpu_ids="0,1",
        api_keys="secret-key-1,secret-key-2",
        chunk_target_sec=10.0,
        chunk_search_expand_sec=1.0,
        chunks_per_dispatch=2,
        max_sync_audio_sec=30.0,
        data_dir=data_dir,
    )
    app = create_app(settings_override=settings)
    with TestClient(app) as client:
        yield client


def test_health_reports_dual_workers(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backend"] == "mock"
    assert len(body["workers"]) == 2
    assert {w["gpu_id"] for w in body["workers"]} == {"0", "1"}
    assert all(w["alive"] for w in body["workers"])


def test_api_key_auth(api_client: TestClient) -> None:
    assert api_client.get("/v1/models").status_code == 401
    assert api_client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401

    ok_bearer = api_client.get("/v1/models", headers={"Authorization": "Bearer secret-key-1"})
    assert ok_bearer.status_code == 200
    assert ok_bearer.json()["data"][0]["id"] == "qwen3-asr-1.7b"

    ok_header = api_client.get("/v1/models", headers={"X-API-Key": "secret-key-2"})
    assert ok_header.status_code == 200


def test_async_long_audio_task_dual_gpu_and_subtitles(api_client: TestClient) -> None:
    headers = {"Authorization": "Bearer secret-key-1"}
    wav_bytes = _make_wav_bytes(35.0)  # Splits into ~4 chunks across both mock GPU workers

    create_resp = api_client.post(
        "/v1/asr/tasks",
        headers=headers,
        files={"file": ("meeting.wav", wav_bytes, "audio/wav")},
        data={"language": "zh", "return_timestamps": "true", "include_words": "true"},
    )
    assert create_resp.status_code == 202
    task_id = create_resp.json()["task_id"]

    # Poll until completed
    final_task = None
    for _ in range(60):
        poll = api_client.get(f"/v1/asr/tasks/{task_id}", headers=headers)
        assert poll.status_code == 200
        data = poll.json()
        if data["status"] in ("completed", "failed"):
            final_task = data
            break
        time.sleep(0.1)

    assert final_task is not None
    assert final_task["status"] == "completed", final_task.get("error")
    assert final_task["progress"] == 1.0
    assert final_task["total_chunks"] >= 3
    assert final_task["completed_chunks"] == final_task["total_chunks"]

    result = final_task["result"]
    assert result["language"] == "Chinese"
    assert abs(result["duration_sec"] - 35.0) < 0.2
    assert len(result["segments"]) >= 6  # 2 sentences per chunk in mock
    assert len(result["words"]) > 0

    # Verify timestamps increase monotonically across chunks
    starts = [s["start"] for s in result["segments"]]
    assert starts == sorted(starts)
    assert result["segments"][-1]["end"] > 30.0

    # Subtitle export (SRT + VTT)
    srt_resp = api_client.get(
        f"/v1/asr/tasks/{task_id}/subtitle", params={"format": "srt"}, headers=headers
    )
    assert srt_resp.status_code == 200
    assert "-->" in srt_resp.text

    vtt_resp = api_client.get(
        f"/v1/asr/tasks/{task_id}/subtitle", params={"format": "vtt"}, headers=headers
    )
    assert vtt_resp.status_code == 200
    assert vtt_resp.text.startswith("WEBVTT")

    # List & Delete
    list_resp = api_client.get("/v1/asr/tasks", headers=headers)
    assert list_resp.status_code == 200
    assert any(t["task_id"] == task_id for t in list_resp.json())

    del_resp = api_client.delete(f"/v1/asr/tasks/{task_id}", headers=headers)
    assert del_resp.status_code == 200
    assert api_client.get(f"/v1/asr/tasks/{task_id}", headers=headers).status_code == 404


def test_webhook_ssrf_guard(api_client: TestClient) -> None:
    headers = {"Authorization": "Bearer secret-key-1"}
    wav_bytes = _make_wav_bytes(2.0)

    for bad_url in ["http://127.0.0.1:9000/cb", "http://10.0.0.5/cb", "file:///etc/passwd"]:
        resp = api_client.post(
            "/v1/asr/tasks",
            headers=headers,
            files={"file": ("a.wav", wav_bytes, "audio/wav")},
            data={"webhook_url": bad_url},
        )
        assert resp.status_code == 400


def test_sync_and_openai_compatible_endpoints(api_client: TestClient) -> None:
    headers = {"Authorization": "Bearer secret-key-1"}
    short_wav = _make_wav_bytes(3.0)

    # Custom sync endpoint
    sync_resp = api_client.post(
        "/v1/asr/transcribe",
        headers=headers,
        files={"file": ("short.wav", short_wav, "audio/wav")},
        data={"language": "en", "return_timestamps": "true"},
    )
    assert sync_resp.status_code == 200
    sync_data = sync_resp.json()
    assert sync_data["language"] == "English"
    assert "Hello world." in sync_data["text"]
    assert len(sync_data["segments"]) == 2
    assert len(sync_data["words"]) == 4

    # OpenAI compatible verbose_json
    oai_resp = api_client.post(
        "/v1/audio/transcriptions",
        headers=headers,
        files={"file": ("short.wav", short_wav, "audio/wav")},
        data={
            "model": "qwen3-asr-1.7b",
            "language": "en",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["segment", "word"],
        },
    )
    assert oai_resp.status_code == 200
    oai_data = oai_resp.json()
    assert oai_data["task"] == "transcribe"
    assert len(oai_data["segments"]) == 2
    assert len(oai_data["words"]) == 4
    assert "word" in oai_data["words"][0]

    # Sync duration guard (> 30s rejected)
    too_long_wav = _make_wav_bytes(35.0)
    guard_resp = api_client.post(
        "/v1/asr/transcribe",
        headers=headers,
        files={"file": ("too_long.wav", too_long_wav, "audio/wav")},
    )
    assert guard_resp.status_code == 400
    assert "/v1/asr/tasks" in guard_resp.json()["detail"]
