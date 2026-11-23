"""Audio upload streaming, ffmpeg transcoding, and low-energy long-audio chunking."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile, status

SAMPLE_RATE = 16000
MIN_CHUNK_SECONDS = 0.5
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024  # 1 MB


@dataclass(frozen=True)
class ChunkSpec:
    """Slice metadata referencing a segment of a 16kHz mono WAV file on disk."""

    chunk_index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


async def save_upload_file(upload: UploadFile, dest_path: Path, max_bytes: int) -> int:
    """Stream an UploadFile to disk while enforcing max_bytes without buffering in RAM."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with dest_path.open("wb") as f:
            while True:
                chunk = await upload.read(UPLOAD_READ_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Uploaded file exceeds limit of {max_bytes // (1024 * 1024)} MB.",
                    )
                f.write(chunk)
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if written == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )
    return written


def _decode_with_soundfile_fallback(src_path: Path, dst_wav_path: Path) -> float:
    """Fallback decoder using soundfile when ffmpeg is unavailable (supports WAV/FLAC/OGG)."""
    audio, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    if audio.size == 0:
        raise ValueError("Decoded audio contains zero samples.")
    if sr != SAMPLE_RATE:
        # Linear interpolation fallback for dev/test environments without ffmpeg.
        new_len = max(1, int(round(len(audio) * SAMPLE_RATE / float(sr))))
        old_idx = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_idx = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(new_idx, old_idx, audio).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(dst_wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
    return round(len(audio) / float(SAMPLE_RATE), 3)


async def decode_to_wav_16k(src_path: Path, dst_wav_path: Path) -> float:
    """Convert any audio/video file to 16kHz mono PCM16 WAV via ffmpeg and return duration (sec)."""
    dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(src_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst_wav_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            dst_wav_path.unlink(missing_ok=True)
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"ffmpeg failed to decode audio: {err_msg or 'unknown error'}")
        info = sf.info(str(dst_wav_path))
        if info.frames <= 0:
            dst_wav_path.unlink(missing_ok=True)
            raise ValueError("Decoded audio has zero duration.")
        return round(info.frames / float(info.samplerate), 3)

    return await asyncio.to_thread(_decode_with_soundfile_fallback, src_path, dst_wav_path)


def compute_low_energy_chunks(
    wav_path: Path,
    target_sec: float = 120.0,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> List[ChunkSpec]:
    """
    Compute non-overlapping, gap-free chunk boundaries for a 16kHz mono WAV file.

    Only reads small ±search_expand_sec windows around candidate split points, keeping
    RAM usage tiny even for multi-hour recordings.
    """
    with sf.SoundFile(str(wav_path), mode="r") as f:
        sr = int(f.samplerate)
        total_frames = int(f.frames)
        if total_frames <= 0:
            return []

        max_len = max(1, int(round(target_sec * sr)))
        if total_frames <= max_len:
            return [
                ChunkSpec(
                    chunk_index=0,
                    start_frame=0,
                    end_frame=total_frames,
                    start_sec=0.0,
                    end_sec=round(total_frames / float(sr), 3),
                )
            ]

        expand = int(round(search_expand_sec * sr))
        win = max(4, int(round((min_window_ms / 1000.0) * sr)))

        chunks: List[ChunkSpec] = []
        start = 0
        chunk_idx = 0

        while (total_frames - start) > max_len:
            cut = start + max_len
            left = max(start, cut - expand)
            right = min(total_frames, cut + expand)

            if (right - left) <= win:
                boundary = cut
            else:
                f.seek(left)
                seg = f.read(frames=right - left, dtype="float32", always_2d=False)
                if seg.ndim == 2:
                    seg = np.mean(seg, axis=-1).astype(np.float32)
                seg_abs = np.abs(seg)
                window_sums = np.convolve(
                    seg_abs, np.ones(win, dtype=np.float32), mode="valid"
                )
                min_pos = int(np.argmin(window_sums))
                local = seg_abs[min_pos : min_pos + win]
                inner = int(np.argmin(local))
                boundary = left + min_pos + inner

            boundary = max(start + 1, min(int(boundary), total_frames))
            chunks.append(
                ChunkSpec(
                    chunk_index=chunk_idx,
                    start_frame=start,
                    end_frame=boundary,
                    start_sec=round(start / float(sr), 3),
                    end_sec=round(boundary / float(sr), 3),
                )
            )
            chunk_idx += 1
            start = boundary

        if start < total_frames:
            chunks.append(
                ChunkSpec(
                    chunk_index=chunk_idx,
                    start_frame=start,
                    end_frame=total_frames,
                    start_sec=round(start / float(sr), 3),
                    end_sec=round(total_frames / float(sr), 3),
                )
            )

        return chunks


def read_chunk_waveform(
    wav_path: Path,
    start_frame: int,
    end_frame: int,
    min_sec: float = MIN_CHUNK_SECONDS,
) -> tuple[np.ndarray, int]:
    """Seek-read a specific frame range as mono float32 waveform in [-1, 1]."""
    with sf.SoundFile(str(wav_path), mode="r") as f:
        sr = int(f.samplerate)
        f.seek(start_frame)
        wav = f.read(frames=max(0, end_frame - start_frame), dtype="float32", always_2d=False)

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = np.mean(wav, axis=-1).astype(np.float32)

    min_frames = int(round(min_sec * sr))
    if wav.shape[0] < min_frames:
        wav = np.pad(wav, (0, min_frames - wav.shape[0]), mode="constant").astype(np.float32)

    peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
    if peak > 1.0:
        wav = wav / peak
    return np.clip(wav, -1.0, 1.0), sr
