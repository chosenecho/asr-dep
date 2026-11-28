"""Modal serverless GPU deployment for qwen3-asr-server (NVIDIA L4 24GB).

Cold-start optimizations enabled:
- Modal CPU + GPU memory snapshot (`enable_memory_snapshot=True`,
  `experimental_options={"enable_gpu_snapshot": True}`) with `@modal.enter(snap=True)`
- Pre-snapshot end-to-end warmup (vLLM ASR + ForcedAligner + Silero VAD ONNX +
  FastAPI/ffmpeg HTTP path) to eliminate post-restore JIT stalls
- Persistent JIT compilation cache Volume (`/cache`) for vLLM, Triton,
  TorchInductor, FlashInfer, and CUDA PTX caches, committed synchronously after warmup
- Trimmed CUDA Graph capture (`FULL_DECODE_ONLY`, sizes `1,2,4,8,16`) with GPU
  snapshots enabled; automatic fallback to `VLLM_ENFORCE_EAGER=true` when disabled
- In-process vLLM V1 engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) inside the
  isolated GPU worker process for faster cold start and cleaner GPU snapshotting
- Single-GPU BF16 weights (~6.2 GB) pre-stored on persistent Modal Volume (`/models`)
- Short scaledown window (60s) for responsive burst traffic at minimal idle cost
- Modal edge proxy authentication (`requires_proxy_auth=True`): rejects unauthorized
  requests at Modal's edge gateway (HTTP 401) without waking GPU containers ($0 compute);
  callers must pass `Modal-Key` + `Modal-Secret` headers in addition to `Authorization: Bearer`

Usage:
    # 1. Pre-download weights into Modal Volume `qwen3-asr-weights` (one-time, ~15s):
    modal run deploy/modal_app.py::download_models

    # 2. Deploy serverless HTTPS endpoint:
    modal deploy deploy/modal_app.py
"""

from __future__ import annotations

import io
import math
import os
import struct
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import modal

APP_NAME = "qwen3-asr-server"
ENDPOINT_LABEL = "qwen3-asr-server-serve"
SERVER_PORT = 8000
REPO_ROOT = Path(__file__).resolve().parent.parent
API_SECRET_NAME = os.environ.get("MODAL_API_SECRET_NAME", "qwen-api-key")

# Cold-start & security knobs
ENABLE_GPU_SNAPSHOT = True
REQUIRES_PROXY_AUTH = True
SCALEDOWN_WINDOW_SECONDS = 60
# Trimmed CUDA Graph sizes matching MAX_INFERENCE_BATCH_SIZE=16 (5 decode graphs vs ~51 default).
CUDAGRAPH_MODE_WITH_SNAPSHOT = "FULL_DECODE_ONLY"
CUDAGRAPH_CAPTURE_SIZES = "1,2,4,8,16"

MODELS = [
    ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B"),
    ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner-0.6B"),
]

# Persistent volumes for weights (~6.2 GB) and vLLM / Triton / TorchInductor / FlashInfer caches
weights_vol = modal.Volume.from_name("qwen3-asr-weights", create_if_missing=True)
cache_vol = modal.Volume.from_name("qwen3-asr-cache", create_if_missing=True)

app = modal.App(APP_NAME)

# Fast lightweight image dedicated to downloading weights from HuggingFace into the Volume
downloader_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub>=0.26.0", "hf-transfer>=0.1.8")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Production inference image (CUDA 12.8 runtime + Python 3.12 + qwen-asr[vllm] + Silero VAD ONNX)
asr_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("ffmpeg", "libsndfile1", "curl", "gcc", "g++")
    .uv_pip_install(
        "qwen-asr[vllm]",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "python-multipart>=0.0.12",
        "pydantic>=2.9.0",
        "pydantic-settings>=2.6.0",
        "httpx>=0.27.0",
        "numpy>=1.26.0",
        "soundfile>=0.12.1",
        "silero-vad>=5.1.2",
        "onnxruntime>=1.18.0",
        "huggingface-hub>=0.26.0",
        "hf-transfer>=0.1.8",
    )
    .env(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "VLLM_WORKER_MULTIPROCESS_METHOD": "spawn",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "HF_HOME": "/cache/huggingface",
            "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
            "TRITON_CACHE_DIR": "/cache/triton",
            "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
            "CUDA_CACHE_PATH": "/cache/nv",
            "XDG_CACHE_HOME": "/cache/xdg",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/app",
        }
    )
    .workdir("/app")
    .add_local_dir(
        str(REPO_ROOT / "app"),
        remote_path="/app/app",
        ignore=["**/__pycache__", "**/*.pyc", "**/*.pyo"],
    )
)


def _ensure_models_present(models_root: Path) -> bool:
    """Download missing model directories from HuggingFace. Returns True if downloaded."""
    from huggingface_hub import snapshot_download

    downloaded = False
    models_root.mkdir(parents=True, exist_ok=True)
    for repo_id, folder in MODELS:
        target = models_root / folder
        config_marker = target / "config.json"
        if config_marker.exists():
            print(f"[models] Already present: {target}")
            continue
        print(f"[models] Downloading {repo_id} -> {target} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
        )
        downloaded = True
        print(f"[models] Completed: {target}")
    return downloaded


@app.function(
    image=downloader_image,
    volumes={"/models": weights_vol},
    timeout=1800,
    cpu=4.0,
    memory=8192,
)
def download_models() -> str:
    """Pre-populate the `qwen3-asr-weights` Modal Volume from HuggingFace."""
    downloaded = _ensure_models_present(Path("/models"))
    if downloaded:
        weights_vol.commit()
    sizes = []
    for _, folder in MODELS:
        p = Path("/models") / folder
        total_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
        sizes.append(f"{folder}: {total_mb:.1f} MiB")
    summary = " | ".join(sizes)
    print(f"[models] Volume ready -> {summary}")
    return summary


def _make_warmup_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a tiny 16kHz mono PCM WAV in memory for pre-snapshot HTTP warmup."""
    n_samples = max(1, int(duration_sec * sample_rate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray(n_samples * 2)
        for i in range(n_samples):
            val = int(1600 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            struct.pack_into("<h", frames, i * 2, val)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def _build_multipart_body(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str = "audio/wav",
) -> tuple[bytes, str]:
    boundary = "----ModalWarmupBoundary7MA4YWxkTrZu0gW"
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
            f"{v}\r\n".encode("utf-8")
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _wait_and_warmup_server(
    proc: subprocess.Popen[bytes],
    api_key: str,
    port: int = SERVER_PORT,
    timeout_sec: float = 500.0,
) -> None:
    """Wait until uvicorn /health returns 200, then run one sync HTTP transcription warmup."""
    t0 = time.monotonic()
    deadline = t0 + timeout_sec
    health_url = f"http://127.0.0.1:{port}/health"

    while True:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"uvicorn exited prematurely during startup (exit code {rc}).")
        if time.monotonic() > deadline:
            raise RuntimeError(f"Timed out after {timeout_sec:.0f}s waiting for {health_url}.")
        try:
            req = urllib.request.Request(health_url, headers={"Connection": "close"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    print(f"[modal] /health ready in {time.monotonic() - t0:.1f}s: {body}")
                    break
        except Exception:
            pass
        time.sleep(1.0)

    # End-to-end synchronous HTTP warmup (no SQLite task row created; closes socket cleanly).
    t_warm = time.monotonic()
    wav_bytes = _make_warmup_wav_bytes(duration_sec=1.0)
    body_bytes, content_type = _build_multipart_body(
        fields={
            "language": "English",
            "return_timestamps": "true",
            "include_words": "true",
            "response_format": "json",
        },
        file_field="file",
        filename="warmup.wav",
        file_bytes=wav_bytes,
    )
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/asr/transcribe",
        data=body_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
            "Connection": "close",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            _ = resp.read()
            print(
                f"[modal] End-to-end HTTP warmup succeeded (status={resp.status}) "
                f"in {time.monotonic() - t_warm:.2f}s."
            )
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"End-to-end warmup HTTP request failed ({exc.code}): {err_body}"
        ) from exc

    # Allow uvicorn to finish tearing down the warmup TCP socket before snapshotting.
    time.sleep(0.5)


@app.cls(
    image=asr_image,
    gpu="L4",
    cpu=8.0,
    memory=24576,
    volumes={
        "/models": weights_vol,
        "/cache": cache_vol,
    },
    secrets=[modal.Secret.from_name(API_SECRET_NAME)],
    enable_memory_snapshot=ENABLE_GPU_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": True} if ENABLE_GPU_SNAPSHOT else None,
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    startup_timeout=600,
    timeout=3600,
)
@modal.concurrent(max_inputs=64)
class Qwen3ASRServer:
    """Serverless Qwen3-ASR-1.7B + ForcedAligner API backed by Modal GPU snapshots."""

    @modal.enter(snap=ENABLE_GPU_SNAPSHOT)
    def start_and_warmup(self) -> None:
        """Start uvicorn + GPU worker, warm all kernels/VAD, and commit /cache before snapshot."""
        t0 = time.monotonic()

        # Map API key from Modal Secret (supports API_KEYS / SGLANG_API_KEY / QWEN_API_KEY / API_KEY) -> API_KEYS
        api_keys = (
            os.environ.get("API_KEYS")
            or os.environ.get("SGLANG_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("API_KEY")
            or ""
        ).strip()
        if not api_keys:
            raise RuntimeError(
                f"No API key found in Modal secret `{API_SECRET_NAME}` "
                "(expected API_KEYS, QWEN_API_KEY, SGLANG_API_KEY, or API_KEY). "
                "Refusing to start unauthenticated."
            )
        first_api_key = next((k.strip() for k in api_keys.split(",") if k.strip()), "")

        # Fallback auto-download in case volume wasn't pre-populated
        if _ensure_models_present(Path("/models")):
            weights_vol.commit()

        for d in (
            "/tmp/qwen3_asr_data",
            "/tmp/qwen3_asr_tmp",
            "/cache/vllm",
            "/cache/huggingface",
            "/cache/torchinductor",
            "/cache/triton",
            "/cache/flashinfer",
            "/cache/nv",
            "/cache/xdg",
        ):
            Path(d).mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "HOST": "0.0.0.0",
                "PORT": str(SERVER_PORT),
                "LOG_LEVEL": "INFO",
                "API_KEYS": api_keys,
                "ASR_MODEL_PATH": "/models/Qwen3-ASR-1.7B",
                "ALIGNER_MODEL_PATH": "/models/Qwen3-ForcedAligner-0.6B",
                "SERVED_MODEL_NAME": "qwen3-asr-1.7b",
                "ASR_BACKEND": "vllm",
                "GPU_IDS": "0",
                # Run vLLM V1 EngineCore in-process inside the dedicated GPU worker process:
                # removes an extra Python/CUDA child process + ZMQ IPC hop for faster boot
                # and cleaner GPU checkpoint/restore.
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                # Persistent JIT caches on /cache Volume (fallback when snapshot is invalidated)
                "VLLM_CACHE_ROOT": "/cache/vllm",
                "HF_HOME": "/cache/huggingface",
                "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
                "TRITON_CACHE_DIR": "/cache/triton",
                "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
                "CUDA_CACHE_PATH": "/cache/nv",
                "XDG_CACHE_HOME": "/cache/xdg",
                # Tuned for NVIDIA L4 (24GB / 22.5 GiB usable VRAM):
                # ~13.0 GiB vLLM (with max_model_len=16384 -> ~5.3 GiB KV cache, ~49k tokens)
                # + ~9.5 GiB headroom for ForcedAligner-0.6B & forward activations
                "GPU_MEMORY_UTILIZATION": "0.58",
                "VLLM_MAX_MODEL_LEN": "16384",
                "MAX_INFERENCE_BATCH_SIZE": "16",
                "ALIGNER_BATCH_SIZE": "6",
                "CHUNKS_PER_DISPATCH": "4",
                "BATCH_COALESCE_WAIT_MS": "25",
                "MAX_NEW_TOKENS": "2048",
                "CHUNK_STRATEGY": "vad",
                "VAD_MAX_WORKERS": "4",
                "MAX_CONCURRENT_DECODES": "4",
                # Ephemeral containers cannot promise durable asynchronous jobs.
                "ENABLE_ASYNC_TASKS": "false",
                "DATA_DIR": "/tmp/qwen3_asr_data",
                "TEMP_DIR": "/tmp/qwen3_asr_tmp",
                "WORKER_STARTUP_TIMEOUT_SEC": "480",
                # Force ASR + ForcedAligner + Silero VAD warmup before /health returns 200
                "WARMUP_ON_STARTUP": "true",
            }
        )

        if ENABLE_GPU_SNAPSHOT:
            # Trimmed decode-only CUDA Graphs: fast capture, compact GPU snapshot, full decode speed
            env["VLLM_ENFORCE_EAGER"] = "false"
            env["VLLM_CUDAGRAPH_MODE"] = CUDAGRAPH_MODE_WITH_SNAPSHOT
            env["VLLM_CUDAGRAPH_CAPTURE_SIZES"] = CUDAGRAPH_CAPTURE_SIZES
        else:
            # Without GPU snapshot, skip CUDA Graph capture & torch.compile for shorter cold starts
            env["VLLM_ENFORCE_EAGER"] = "true"

        cmd = [
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(SERVER_PORT),
            "--workers",
            "1",
            "--proxy-headers",
        ]
        self._proc = subprocess.Popen(cmd, cwd="/app", env=env)

        _wait_and_warmup_server(self._proc, api_key=first_api_key, port=SERVER_PORT)

        # Persist JIT compilation caches synchronously before Modal freezes the container snapshot
        try:
            cache_vol.commit()
            print("[modal] Committed /cache volume synchronously before snapshot.")
        except Exception as exc:
            print(f"[modal] Warning: failed to commit /cache volume ({exc}); continuing.")

        print(
            f"[modal] Pre-snapshot startup & warmup completed in {time.monotonic() - t0:.1f}s "
            f"(gpu_snapshot={ENABLE_GPU_SNAPSHOT}, scaledown={SCALEDOWN_WINDOW_SECONDS}s)."
        )

    @modal.web_server(
        port=SERVER_PORT,
        startup_timeout=300,
        label=ENDPOINT_LABEL,
        requires_proxy_auth=REQUIRES_PROXY_AUTH,
    )
    def serve(self) -> None:
        """Expose the already-warmed uvicorn server on port 8000 (restored via GPU snapshot)."""
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"uvicorn process is not running after snapshot restore (exit code {proc.returncode})."
            )

    @modal.exit()
    def shutdown(self) -> None:
        """Gracefully stop uvicorn and its GPU worker subprocess on container scale-down."""
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
