"""Own staged audio, decode concurrency, and storage until the last reader drains."""
from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from .admission import AdmissionController, AdmissionTicket
from .audio import SAMPLE_RATE, decode_to_wav_16k, save_upload_file


@dataclass
class PreparedAudio:
    request_id: str
    raw_path: Path
    wav_path: Path
    ticket: AdmissionTicket
    duration_sec: float = 0.0
    source_bytes: int = 0
    decoded_bytes: int = 0
    _reserved_bytes: int = 0
    _released: bool = False

    @property
    def actual_bytes(self) -> int:
        return self.source_bytes + self.decoded_bytes


class AudioPreparation:
    def __init__(self, settings: Any, admission: AdmissionController):
        self.settings = settings
        self.admission = admission
        self._decode_slots = asyncio.Semaphore(settings.max_concurrent_decodes)
        self.decoding_jobs = 0

    async def stage(self, upload: UploadFile, request_id: str,
                    ticket: AdmissionTicket) -> PreparedAudio:
        directory = self.settings.temp_audio_dir / ("job_" + uuid.uuid4().hex)
        directory.mkdir(parents=True, exist_ok=False)
        prepared = PreparedAudio(request_id, directory / "audio.raw", directory / "audio.wav", ticket)

        def reserve_chunk(size: int) -> None:
            ticket.reserve_bytes(size)
            prepared._reserved_bytes += size

        try:
            prepared.source_bytes = await save_upload_file(
                upload, prepared.raw_path, self.settings.max_upload_bytes, on_chunk=reserve_chunk,
            )
            return prepared
        except BaseException:
            await self.release(prepared)
            raise

    async def decode(self, prepared: PreparedAudio, max_duration_sec: float) -> float:
        if prepared._released:
            raise RuntimeError("Prepared audio has already been released.")
        if prepared.decoded_bytes:
            if prepared.duration_sec > max_duration_sec:
                raise ValueError("Audio duration exceeds limit.")
            return prepared.duration_sec
        async with self._decode_slots:
            # Include ffmpeg's detection allowance and ample WAV header overhead.
            budget = min(int(self.settings.max_decoded_audio_mb * 1024 * 1024),
                         math.ceil((max_duration_sec + 0.25) * SAMPLE_RATE * 2) + 65536)
            prepared.ticket.reserve_bytes(budget)
            prepared._reserved_bytes += budget
            self.decoding_jobs += 1
            try:
                duration = await decode_to_wav_16k(
                    prepared.raw_path, prepared.wav_path,
                    timeout_sec=self.settings.decode_timeout_sec,
                    max_duration_sec=max_duration_sec, max_output_bytes=budget,
                    allow_soundfile_fallback=self.settings.asr_backend == "mock",
                )
                actual = prepared.wav_path.stat().st_size
                if actual > budget:
                    raise ValueError("Decoded audio exceeds output byte limit.")
                prepared.ticket.release_bytes(budget - actual)
                prepared._reserved_bytes -= budget - actual
                prepared.decoded_bytes = actual
                prepared.duration_sec = duration
                return duration
            except BaseException:
                prepared.wav_path.unlink(missing_ok=True)
                prepared.ticket.release_bytes(budget)
                prepared._reserved_bytes -= budget
                raise
            finally:
                self.decoding_jobs -= 1

    async def release(self, prepared: PreparedAudio) -> None:
        """Caller must first await dispatcher drain; does not release the job ticket."""
        if prepared._released:
            return
        prepared.raw_path.unlink(missing_ok=True)
        prepared.wav_path.unlink(missing_ok=True)
        prepared.raw_path.parent.rmdir()
        prepared.ticket.release_bytes(prepared._reserved_bytes)
        prepared._reserved_bytes = 0
        prepared._released = True
