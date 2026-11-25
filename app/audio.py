"""Audio upload streaming, ffmpeg transcoding, and VAD / low-energy long-audio chunking."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile, status

logger = logging.getLogger("qwen3_asr.audio")

SAMPLE_RATE = 16000
MIN_CHUNK_SECONDS = 0.5
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024  # 1 MB

_VAD_INIT_LOCK = threading.Lock()
_VAD_TLS = threading.local()
_VAD_INIT_FAILED = False
_VAD_WARNED = False
_VAD_LOGGED_BACKEND = False
_VAD_EXECUTOR_LOCK = threading.Lock()
_VAD_EXECUTOR: Optional[ThreadPoolExecutor] = None
_VAD_EXECUTOR_WORKERS = 0


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
            "-threads",
            "2",
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


def normalize_for_vad(
    seg: np.ndarray,
    target_rms: float = 0.08,
    min_rms: float = 1e-5,
    min_gain: float = 0.1,
    max_gain: float = 30.0,
) -> np.ndarray:
    """
    Remove DC offset and apply bounded RMS normalization on a temporary copy
    for Silero VAD boundary detection.

    Ensures both very quiet recordings (e.g. -45 dBFS) and very loud recordings
    fall into Silero VAD's optimal operating loudness without amplifying pure
    digital silence into false speech. Never mutates the WAV sent to ASR.
    """
    x = np.asarray(seg, dtype=np.float32)
    if x.size == 0:
        return x
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    if rms < min_rms or peak < 1e-4:
        return x
    gain = float(np.clip(target_rms / rms, min_gain, max_gain))
    y = x * gain
    new_peak = float(np.max(np.abs(y)))
    if new_peak > 0.98:
        y = y * (0.98 / new_peak)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _get_silero_vad_model() -> Any:
    """Lazily load a thread-local CPU ONNX Silero VAD session (zero GPU VRAM, lock-free concurrency)."""
    global _VAD_INIT_FAILED, _VAD_WARNED, _VAD_LOGGED_BACKEND
    cached = getattr(_VAD_TLS, "model", None)
    if cached is not None:
        return cached
    if _VAD_INIT_FAILED:
        return None
    try:
        from silero_vad import load_silero_vad

        try:
            model = load_silero_vad(onnx=True)
            backend_desc = "ONNX CPU"
        except Exception:
            model = load_silero_vad(onnx=False)
            backend_desc = "PyTorch CPU"
        _VAD_TLS.model = model
        with _VAD_INIT_LOCK:
            if not _VAD_LOGGED_BACKEND:
                _VAD_LOGGED_BACKEND = True
                logger.info(
                    "Loaded Silero VAD (%s, thread-local) for lossless boundary chunking.",
                    backend_desc,
                )
        return model
    except Exception as exc:
        with _VAD_INIT_LOCK:
            _VAD_INIT_FAILED = True
            if not _VAD_WARNED:
                _VAD_WARNED = True
                logger.warning(
                    "Silero VAD unavailable (%s); falling back to low-energy boundary chunking.",
                    exc,
                )
        return None


def _get_vad_executor(max_workers: int) -> ThreadPoolExecutor:
    """Return a shared bounded ThreadPoolExecutor for parallel Silero VAD window evaluation."""
    global _VAD_EXECUTOR, _VAD_EXECUTOR_WORKERS
    workers = max(1, int(max_workers))
    with _VAD_EXECUTOR_LOCK:
        if _VAD_EXECUTOR is None or _VAD_EXECUTOR_WORKERS != workers:
            _VAD_EXECUTOR = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="silero-vad",
            )
            _VAD_EXECUTOR_WORKERS = workers
        return _VAD_EXECUTOR


def _detect_speech_with_silero(
    seg_vad: np.ndarray,
    sr: int,
    threshold: float = 0.5,
    min_silence_ms: int = 150,
) -> Optional[List[Dict[str, int]]]:
    """Run Silero VAD on a normalized 16kHz window using the calling thread's dedicated session."""
    model = _get_silero_vad_model()
    if model is None:
        return None
    import torch
    from silero_vad import get_speech_timestamps

    tensor = torch.from_numpy(np.ascontiguousarray(seg_vad, dtype=np.float32))
    if hasattr(model, "reset_states"):
        model.reset_states()
    return get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sr,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=60,
        return_seconds=False,
    )


def _find_low_energy_offset(seg: np.ndarray, win: int) -> int:
    """Return frame offset inside `seg` with lowest sliding-window energy."""
    n = len(seg)
    if n <= 1:
        return 0
    seg_abs = np.abs(seg)
    if n <= win:
        return int(np.argmin(seg_abs))
    window_sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")
    min_pos = int(np.argmin(window_sums))
    local = seg_abs[min_pos : min_pos + win]
    inner = int(np.argmin(local))
    return min_pos + inner


def _pick_vad_boundary_offset(
    seg: np.ndarray,
    sr: int,
    target_rel: int,
    win: int,
    speech_timestamps: List[Dict[str, int]],
    min_gap_ms: int = 120,
) -> Optional[int]:
    """
    Pick the optimal non-speech pause midpoint inside `seg` from Silero VAD speech intervals.
    Returns None if no pause >= min_gap_ms exists (caller falls back to energy minimum).
    """
    n = len(seg)
    if not speech_timestamps:
        # Entire window has no speech (silence or pure music): pick low-energy point near center.
        half = max(1, n // 4)
        lo = max(0, target_rel - half)
        hi = min(n, target_rel + half)
        return lo + _find_low_energy_offset(seg[lo:hi], min(win, max(4, hi - lo)))

    # Merge/clamp speech intervals and compute complementary non-speech gaps.
    intervals: List[tuple[int, int]] = []
    for item in speech_timestamps:
        s = max(0, min(n, int(item["start"])))
        e = max(s, min(n, int(item["end"])))
        if intervals and s <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], e))
        else:
            intervals.append((s, e))

    gaps: List[tuple[int, int]] = []
    cursor = 0
    for s, e in intervals:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < n:
        gaps.append((cursor, n))

    min_gap_frames = max(1, int(round((min_gap_ms / 1000.0) * sr)))
    valid_gaps = [(gs, ge) for gs, ge in gaps if (ge - gs) >= min_gap_frames]
    if not valid_gaps:
        return None

    half_span = max(1.0, n / 2.0)
    best_gap = max(
        valid_gaps,
        key=lambda g: (
            min((g[1] - g[0]) / float(sr), 1.5)
            - 0.15 * (abs(((g[0] + g[1]) * 0.5) - target_rel) / half_span)
        ),
    )
    gs, ge = best_gap
    gap_len = ge - gs
    # Stay inside the central 60% of the pause so we remain far from both adjacent words.
    margin = int(gap_len * 0.2)
    inner_start = gs + margin
    inner_end = max(inner_start + 1, ge - margin)
    sub_win = min(win, max(4, inner_end - inner_start))
    return inner_start + _find_low_energy_offset(seg[inner_start:inner_end], sub_win)


def _resolve_window_boundary(
    seg: np.ndarray,
    sr: int,
    left: int,
    cut: int,
    win: int,
    use_vad: bool,
    vad_threshold: float,
    vad_min_silence_ms: int,
    speech_timestamps_fn: Optional[Callable[[np.ndarray, int], Optional[List[Dict[str, int]]]]],
) -> int:
    """Resolve optimal cut frame inside [left, left + len(seg)] via Silero VAD or energy fallback."""
    if len(seg) <= win:
        return cut

    chosen_rel: Optional[int] = None
    if use_vad:
        try:
            seg_vad = normalize_for_vad(seg)
            if speech_timestamps_fn is not None:
                ts = speech_timestamps_fn(seg_vad, sr)
            else:
                ts = _detect_speech_with_silero(
                    seg_vad,
                    sr,
                    threshold=vad_threshold,
                    min_silence_ms=vad_min_silence_ms,
                )
            if ts is not None:
                chosen_rel = _pick_vad_boundary_offset(
                    seg=seg,
                    sr=sr,
                    target_rel=cut - left,
                    win=win,
                    speech_timestamps=ts,
                    min_gap_ms=min(120, vad_min_silence_ms),
                )
        except Exception as exc:
            logger.warning("Silero VAD boundary search failed (%s); using energy fallback.", exc)

    if chosen_rel is None:
        chosen_rel = _find_low_energy_offset(seg, win)

    return left + int(chosen_rel)


def compute_chunks(
    wav_path: Path,
    target_sec: float = 120.0,
    search_expand_sec: float = 15.0,
    min_window_ms: float = 100.0,
    strategy: str = "vad",
    vad_threshold: float = 0.5,
    vad_min_silence_ms: int = 150,
    speech_timestamps_fn: Optional[Callable[[np.ndarray, int], Optional[List[Dict[str, int]]]]] = None,
    max_workers: int = 1,
) -> List[ChunkSpec]:
    """
    Compute non-overlapping, gap-free (lossless) chunk boundaries for a 16kHz mono WAV.

    - `strategy="vad"`: runs loudness-normalized Silero VAD only on small ±search_expand_sec
      windows around candidate split points, snapping cuts to the middle of real non-speech
      pauses while preserving 100% of audio frames (zero dropped speech/silence).
      When candidate windows are disjoint and `target_sec + 2*search_expand_sec <= 170s`,
      evaluates all windows in parallel across thread-local ONNX sessions for multi-hour speed.
    - `strategy="energy"`: uses sliding-window minimum amplitude around candidate split points.
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
        use_vad = strategy == "vad"

        # Fast parallel disjoint-window mode for multi-chunk recordings:
        # Nominal cut centers at k*max_len (k=1,2,...) have disjoint ±expand windows when
        # max_len > 2*expand + win, guaranteeing worst-case chunk <= target_sec + 2*expand <= 170s.
        can_parallel = (
            max_len > (2 * expand + win)
            and (target_sec + 2.0 * search_expand_sec) <= 170.0
        )
        if can_parallel:
            window_jobs: List[tuple[int, int, np.ndarray]] = []
            k = 1
            while (k * max_len + expand) < total_frames:
                cut = k * max_len
                left = max(0, cut - expand)
                right = min(total_frames, cut + expand)
                f.seek(left)
                seg = f.read(frames=right - left, dtype="float32", always_2d=False)
                if seg.ndim == 2:
                    seg = np.mean(seg, axis=-1).astype(np.float32)
                window_jobs.append((left, cut, seg))
                k += 1

            if window_jobs:
                def _eval_one(job: tuple[int, int, np.ndarray]) -> int:
                    w_left, w_cut, w_seg = job
                    return _resolve_window_boundary(
                        seg=w_seg,
                        sr=sr,
                        left=w_left,
                        cut=w_cut,
                        win=win,
                        use_vad=use_vad,
                        vad_threshold=vad_threshold,
                        vad_min_silence_ms=vad_min_silence_ms,
                        speech_timestamps_fn=speech_timestamps_fn,
                    )

                if use_vad and speech_timestamps_fn is None and max_workers > 1 and len(window_jobs) > 1:
                    executor = _get_vad_executor(max_workers)
                    boundaries = list(executor.map(_eval_one, window_jobs))
                else:
                    boundaries = [_eval_one(j) for j in window_jobs]

                chunks: List[ChunkSpec] = []
                start = 0
                chunk_idx = 0
                for b in boundaries:
                    boundary = max(start + 1, min(int(b), total_frames - 1))
                    if boundary <= start:
                        continue
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

            # Audio is between max_len and max_len + expand (<= 135s <= 170s): keep single chunk
            return [
                ChunkSpec(
                    chunk_index=0,
                    start_frame=0,
                    end_frame=total_frames,
                    start_sec=0.0,
                    end_sec=round(total_frames / float(sr), 3),
                )
            ]

        # Fallback sequential mode (when expand is large relative to target_sec)
        chunks = []
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
                boundary = _resolve_window_boundary(
                    seg=seg,
                    sr=sr,
                    left=left,
                    cut=cut,
                    win=win,
                    use_vad=use_vad,
                    vad_threshold=vad_threshold,
                    vad_min_silence_ms=vad_min_silence_ms,
                    speech_timestamps_fn=speech_timestamps_fn,
                )

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


def compute_low_energy_chunks(
    wav_path: Path,
    target_sec: float = 120.0,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> List[ChunkSpec]:
    """Backward-compatible wrapper for energy-only low-energy chunk boundary computation."""
    return compute_chunks(
        wav_path=wav_path,
        target_sec=target_sec,
        search_expand_sec=search_expand_sec,
        min_window_ms=min_window_ms,
        strategy="energy",
    )


def _normalize_chunk_array(wav: np.ndarray, sr: int, min_sec: float) -> tuple[np.ndarray, int]:
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
    return _normalize_chunk_array(wav, sr, min_sec)


def read_chunks_from_wav(
    wav_path: Path,
    chunks: Sequence[Dict[str, Any]],
    min_sec: float = MIN_CHUNK_SECONDS,
) -> List[tuple[np.ndarray, int]]:
    """Open a WAV file once and seek-read multiple chunk frame ranges."""
    results: List[tuple[np.ndarray, int]] = []
    with sf.SoundFile(str(wav_path), mode="r") as f:
        sr = int(f.samplerate)
        for ch in chunks:
            start_frame = int(ch["start_frame"])
            end_frame = int(ch["end_frame"])
            f.seek(start_frame)
            wav = f.read(frames=max(0, end_frame - start_frame), dtype="float32", always_2d=False)
            results.append(_normalize_chunk_array(wav, sr, min_sec))
    return results
