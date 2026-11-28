"""Pickle-safe IPC. Attempts stay at one; batches are never replayed across workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BatchRequest:
    batch_id: str
    job_id: str
    wav_path: str
    chunks: list[dict[str, Any]]
    language: str | None = None
    context: str = ""
    return_timestamps: bool = True
    attempt: int = 1
    worker_generation: int = 0


@dataclass(frozen=True)
class BatchStarted:
    batch_id: str
    attempt: int
    gpu_id: str
    worker_generation: int


@dataclass(frozen=True)
class BatchResult:
    batch_id: str
    attempt: int
    gpu_id: str
    worker_generation: int
    ok: bool = True
    chunks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class BatchSkipped:
    batch_id: str
    attempt: int
    gpu_id: str
    worker_generation: int


@dataclass(frozen=True)
class PrefetchFailed:
    error: str
