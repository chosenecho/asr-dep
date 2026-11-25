"""Unit tests for low-energy long-audio chunking, timestamp segment builder, and SRT/VTT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from app.alignment import (
    WordTimestamp,
    build_segments_from_chunk,
    format_srt,
    format_vtt,
    has_alignable_text,
    merge_languages,
)
from app.audio import (
    SAMPLE_RATE,
    compute_chunks,
    compute_low_energy_chunks,
    normalize_for_vad,
    read_chunk_waveform,
    read_chunks_from_wav,
)
from app.worker import normalize_requested_language


def test_low_energy_chunking_snaps_to_silence_and_preserves_all_frames(tmp_path: Path) -> None:
    # 30s audio: loud sine wave everywhere EXCEPT a silent valley at 10.6s..11.2s and 20.8s..21.4s
    duration_sec = 30
    total_frames = duration_sec * SAMPLE_RATE
    t = np.arange(total_frames, dtype=np.float32) / SAMPLE_RATE
    wav = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Carve silence near candidate cut points (target_sec=10, search_expand_sec=2)
    wav[int(10.6 * SAMPLE_RATE) : int(11.2 * SAMPLE_RATE)] = 0.0
    wav[int(20.8 * SAMPLE_RATE) : int(21.4 * SAMPLE_RATE)] = 0.0

    wav_path = tmp_path / "long_test.wav"
    sf.write(str(wav_path), wav, SAMPLE_RATE, subtype="PCM_16")

    chunks = compute_low_energy_chunks(wav_path, target_sec=10.0, search_expand_sec=2.0)
    assert len(chunks) == 3

    # Exact frame continuity (no gaps, no overlaps)
    assert chunks[0].start_frame == 0
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.end_frame == cur.start_frame
    assert chunks[-1].end_frame == total_frames

    # First boundary should snap inside the silent valley [10.6, 11.2]
    assert 10.55 <= chunks[0].end_sec <= 11.25


def test_normalize_for_vad_handles_very_quiet_loud_and_silence() -> None:
    t = np.arange(SAMPLE_RATE * 2, dtype=np.float32) / SAMPLE_RATE
    base = np.sin(2 * np.pi * 300.0 * t).astype(np.float32)

    # Very quiet speech (~ -45 dBFS) should be boosted near target_rms (0.08)
    quiet = base * 0.005
    quiet_norm = normalize_for_vad(quiet)
    quiet_rms = float(np.sqrt(np.mean(quiet_norm * quiet_norm)))
    assert 0.06 <= quiet_rms <= 0.10

    # Very loud speech (peak ~0.98) should be attenuated near target_rms without clipping
    loud = base * 0.98
    loud_norm = normalize_for_vad(loud)
    loud_rms = float(np.sqrt(np.mean(loud_norm * loud_norm)))
    assert 0.06 <= loud_rms <= 0.10
    assert float(np.max(np.abs(loud_norm))) <= 0.99

    # Pure near-zero silence must NOT be amplified into false speech
    silence = np.full(SAMPLE_RATE, 1e-6, dtype=np.float32)
    sil_norm = normalize_for_vad(silence)
    assert float(np.max(np.abs(sil_norm))) < 1e-4


def test_vad_guided_chunking_snaps_to_longest_pause_under_bgm_and_stays_lossless(tmp_path: Path) -> None:
    # 30s audio with constant loud BGM so energy alone cannot distinguish speech vs pause
    duration_sec = 30
    total_frames = duration_sec * SAMPLE_RATE
    t = np.arange(total_frames, dtype=np.float32) / SAMPLE_RATE
    wav = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    wav_path = tmp_path / "bgm_test.wav"
    sf.write(str(wav_path), wav, SAMPLE_RATE, subtype="PCM_16")

    # Mock VAD detector: speech across window except a clear 1.0s non-speech pause at [2.5s, 3.5s]
    # relative to each 4s window (cut=10s, expand=2s -> window [8s, 12s] -> pause at [10.5s, 11.5s])
    def fake_vad(seg_vad: np.ndarray, sr: int):
        assert 0.05 <= float(np.sqrt(np.mean(seg_vad * seg_vad))) <= 0.12
        return [
            {"start": 0, "end": int(2.5 * sr)},
            {"start": int(3.5 * sr), "end": len(seg_vad)},
        ]

    chunks = compute_chunks(
        wav_path,
        target_sec=10.0,
        search_expand_sec=2.0,
        strategy="vad",
        speech_timestamps_fn=fake_vad,
    )
    assert len(chunks) == 3
    assert chunks[0].start_frame == 0
    for prev, cur in zip(chunks, chunks[1:]):
        assert prev.end_frame == cur.start_frame
    assert chunks[-1].end_frame == total_frames
    # Should land inside the central 60% of the [10.5s, 11.5s] non-speech gap -> [10.7s, 11.3s]
    assert 10.65 <= chunks[0].end_sec <= 11.35


def test_read_chunk_waveform_pads_short_tail(tmp_path: Path) -> None:
    wav_path = tmp_path / "short.wav"
    sf.write(str(wav_path), np.ones(1600, dtype=np.float32) * 0.2, SAMPLE_RATE)
    wav, sr = read_chunk_waveform(wav_path, 0, 1600, min_sec=0.5)
    assert sr == SAMPLE_RATE
    assert wav.shape[0] == int(0.5 * SAMPLE_RATE)


def test_read_chunks_from_wav_matches_single_reads(tmp_path: Path) -> None:
    wav_path = tmp_path / "multi.wav"
    t = np.arange(SAMPLE_RATE * 4, dtype=np.float32) / SAMPLE_RATE
    sig = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(wav_path), sig, SAMPLE_RATE, subtype="PCM_16")

    specs = [
        {"start_frame": 0, "end_frame": SAMPLE_RATE * 2},
        {"start_frame": SAMPLE_RATE * 2, "end_frame": SAMPLE_RATE * 4},
    ]
    batch = read_chunks_from_wav(wav_path, specs)
    assert len(batch) == 2
    for sp, (arr, sr) in zip(specs, batch):
        ref, ref_sr = read_chunk_waveform(wav_path, sp["start_frame"], sp["end_frame"])
        assert sr == ref_sr == SAMPLE_RATE
        assert np.allclose(arr, ref)


def test_build_segments_chinese_punctuation_and_words() -> None:
    text = "今天开会。讨论双卡部署！"
    chars = ["今", "天", "开", "会", "讨", "论", "双", "卡", "部", "署"]
    words = [
        WordTimestamp(text=c, start=round(i * 0.5, 3), end=round(i * 0.5 + 0.4, 3))
        for i, c in enumerate(chars)
    ]
    segs = build_segments_from_chunk(text, words, chunk_start=0.0, chunk_end=6.0)
    assert len(segs) == 2
    assert segs[0].text == "今天开会。"
    assert segs[0].start == 0.0
    assert segs[0].end == 1.9
    assert len(segs[0].words) == 4
    assert segs[1].text == "讨论双卡部署！"
    assert segs[1].start == 2.0
    assert segs[1].end == 4.9
    assert len(segs[1].words) == 6


def test_build_segments_english_preserves_decimal_and_splits_on_pause() -> None:
    text = "Pi is 3.14 today Next sentence."
    words = [
        WordTimestamp("Pi", 0.0, 0.3),
        WordTimestamp("is", 0.3, 0.5),
        WordTimestamp("3", 0.5, 0.7),
        WordTimestamp("14", 0.7, 1.0),
        WordTimestamp("today", 1.0, 1.4),
        # 2.0s pause before "Next"
        WordTimestamp("Next", 3.4, 3.8),
        WordTimestamp("sentence", 3.8, 4.4),
    ]
    segs = build_segments_from_chunk(
        text, words, chunk_start=0.0, chunk_end=5.0, pause_threshold_sec=1.2
    )
    assert len(segs) == 2
    assert segs[0].text == "Pi is 3.14 today"
    assert segs[0].end == 1.4
    assert segs[1].text == "Next sentence."
    assert segs[1].start == 3.4
    assert segs[1].end == 4.4


def test_build_segments_fallback_without_words() -> None:
    segs = build_segments_from_chunk("无时间戳回退。", [], chunk_start=12.5, chunk_end=18.2)
    assert len(segs) == 1
    assert segs[0].start == 12.5
    assert segs[0].end == 18.2
    assert segs[0].text == "无时间戳回退。"


def test_srt_and_vtt_formatting() -> None:
    segs = [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "第一句。"},
        {"id": 1, "start": 3661.25, "end": 3665.0, "text": "超过一小时。"},
    ]
    srt = format_srt(segs)
    assert "1\n00:00:00,000 --> 00:00:01,500\n第一句。" in srt
    assert "2\n01:01:01,250 --> 01:01:05,000\n超过一小时。" in srt

    vtt = format_vtt(segs)
    assert vtt.startswith("WEBVTT\n")
    assert "01:01:01.250 --> 01:01:05.000\n超过一小时。" in vtt


def test_helpers() -> None:
    assert has_alignable_text("你好") is True
    assert has_alignable_text("...！？ ") is False
    assert merge_languages(["Chinese", "Chinese", "", "English", "English"]) == "Chinese,English"
    assert normalize_requested_language("zh") == "Chinese"
    assert normalize_requested_language("en") == "English"
    assert normalize_requested_language("auto") is None
