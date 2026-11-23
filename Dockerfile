# Production Dockerfile for Dual RTX 4090 Qwen3-ASR-1.7B + ForcedAligner API Service
# Uses the official qwenllm/qwen3-asr image (pre-bundled with CUDA 12.8, qwen-asr[vllm], flash-attn, ffmpeg).

ARG BASE_IMAGE=qwenllm/qwen3-asr:latest
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

# Ensure ffmpeg & curl (for Docker healthcheck) are available
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY scripts /app/scripts

RUN mkdir -p /app/data /models

EXPOSE 8000

# IMPORTANT: Keep uvicorn --workers 1.
# DualGPUDispatcher inside app/dispatcher.py spawns one isolated process per GPU (GPU_IDS=0,1).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
