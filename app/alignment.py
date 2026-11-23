"""Timestamp segment assembly (punctuated sentences + word timestamps) and SRT/VTT formatting."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

STRONG_BREAK_CHARS = set("。！？!?\n")
SOFT_BREAK_CHARS = set("，、；：,;:")


@dataclass
class WordTimestamp:
    """Single word or CJK character timestamp span (seconds)."""

    text: str
    start: float
    end: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }


@dataclass
class SegmentItem:
    """Sentence/clause-level transcript segment with timestamps and optional words."""

    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = field(default_factory=list)

    def to_dict(self, include_words: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }
        if include_words:
            out["words"] = [w.to_dict() for w in self.words]
        return out


def is_alignable_char(ch: str) -> bool:
    """Matches Qwen3ForceAlignProcessor.is_kept_char (letters, numbers, apostrophe)."""
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def has_alignable_text(text: str) -> bool:
    """Return True if text contains at least one character accepted by Qwen3-ForcedAligner."""
    return any(is_alignable_char(ch) for ch in (text or ""))


def _is_sentence_end_dot(text: str, idx: int) -> bool:
    """Treat ASCII '.' as sentence end unless it looks like a decimal number (e.g. 3.14)."""
    if text[idx] != ".":
        return False
    prev_is_digit = idx > 0 and text[idx - 1].isdigit()
    next_is_digit = (idx + 1) < len(text) and text[idx + 1].isdigit()
    return not (prev_is_digit and next_is_digit)


def build_segments_from_chunk(
    chunk_text: str,
    words: Sequence[WordTimestamp],
    chunk_start: float,
    chunk_end: float,
    start_segment_id: int = 0,
    max_segment_chars: int = 45,
    pause_threshold_sec: float = 1.2,
) -> List[SegmentItem]:
    """
    Reconstruct punctuated sentence/clause segments from chunk ASR text and word timestamps.

    Falls back to a single chunk-level segment when word timestamps are unavailable.
    """
    text = (chunk_text or "").strip()
    if not text:
        return []

    if not words:
        return [
            SegmentItem(
                id=start_segment_id,
                start=round(chunk_start, 3),
                end=round(chunk_end, 3),
                text=text,
                words=[],
            )
        ]

    # Map each alignable character in `text` to the index of its corresponding word token.
    word_norms = ["".join(ch.lower() for ch in w.text if is_alignable_char(ch)) for w in words]
    char_to_word: List[int | None] = [None] * len(text)

    w_idx = 0
    w_char_pos = 0
    for t_idx, ch in enumerate(text):
        if not is_alignable_char(ch):
            continue
        low = ch.lower()

        # Skip any empty normalized words
        while w_idx < len(word_norms) and not word_norms[w_idx]:
            w_idx += 1
            w_char_pos = 0
        if w_idx >= len(word_norms):
            break

        cur_norm = word_norms[w_idx]
        if w_char_pos < len(cur_norm) and cur_norm[w_char_pos] == low:
            char_to_word[t_idx] = w_idx
            w_char_pos += 1
            if w_char_pos >= len(cur_norm):
                w_idx += 1
                w_char_pos = 0
        else:
            # Small lookahead window in case of slight normalization mismatch
            matched = False
            for lookahead in range(w_idx + 1, min(w_idx + 5, len(word_norms))):
                if word_norms[lookahead] and word_norms[lookahead][0] == low:
                    w_idx = lookahead
                    char_to_word[t_idx] = w_idx
                    w_char_pos = 1
                    if w_char_pos >= len(word_norms[w_idx]):
                        w_idx += 1
                        w_char_pos = 0
                    matched = True
                    break
            if not matched:
                char_to_word[t_idx] = w_idx

    segments: List[SegmentItem] = []
    seg_id = start_segment_id
    seg_start_char = 0
    last_word_idx: int | None = None

    def flush_segment(end_char: int) -> None:
        nonlocal seg_id, seg_start_char
        raw_seg = text[seg_start_char:end_char].strip()
        if not raw_seg:
            seg_start_char = end_char
            return

        mapped_indices = [
            wi for wi in char_to_word[seg_start_char:end_char] if wi is not None
        ]
        if mapped_indices:
            first_wi = min(mapped_indices)
            last_wi = max(mapped_indices)
            seg_words = list(words[first_wi : last_wi + 1])
            s_time = seg_words[0].start
            e_time = seg_words[-1].end
        else:
            seg_words = []
            s_time = segments[-1].end if segments else chunk_start
            e_time = max(s_time, chunk_end)

        segments.append(
            SegmentItem(
                id=seg_id,
                start=round(s_time, 3),
                end=round(e_time, 3),
                text=raw_seg,
                words=seg_words,
            )
        )
        seg_id += 1
        seg_start_char = end_char

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        cur_wi = char_to_word[i]

        # Pause-based boundary BEFORE adding the next word
        if (
            cur_wi is not None
            and last_word_idx is not None
            and cur_wi > last_word_idx
            and i > seg_start_char
        ):
            gap = words[cur_wi].start - words[last_word_idx].end
            if gap >= pause_threshold_sec:
                flush_segment(i)

        if cur_wi is not None:
            last_word_idx = cur_wi

        seg_len = i - seg_start_char + 1
        is_strong = ch in STRONG_BREAK_CHARS or _is_sentence_end_dot(text, i)
        is_soft = ch in SOFT_BREAK_CHARS and seg_len >= max_segment_chars

        if is_strong or is_soft:
            # Consume trailing closing quotes/brackets attached to punctuation
            end_pos = i + 1
            while end_pos < n and text[end_pos] in {'"', "'", "”", "’", "）", ")", "」", "』"}:
                end_pos += 1
            flush_segment(end_pos)
            i = end_pos
            continue

        i += 1

    if seg_start_char < n:
        flush_segment(n)

    return segments


def _format_ts(seconds: float, decimal_sep: str) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours = ms_total // 3_600_000
    ms_total %= 3_600_000
    minutes = ms_total // 60_000
    ms_total %= 60_000
    secs = ms_total // 1000
    ms = ms_total % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_sep}{ms:03d}"


def format_srt(segments: Sequence[Dict[str, object]]) -> str:
    """Format segment dicts into standard SubRip (.srt) subtitle text."""
    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = _format_ts(float(seg["start"]), ",")
        end = _format_ts(float(seg["end"]), ",")
        text = str(seg["text"]).strip()
        if not text:
            continue
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + ("\n" if lines else "")


def format_vtt(segments: Sequence[Dict[str, object]]) -> str:
    """Format segment dicts into standard WebVTT (.vtt) subtitle text."""
    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _format_ts(float(seg["start"]), ".")
        end = _format_ts(float(seg["end"]), ".")
        text = str(seg["text"]).strip()
        if not text:
            continue
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"


def merge_languages(langs: Sequence[str]) -> str:
    """Merge per-chunk language names, preserving order and dropping consecutive duplicates."""
    out: List[str] = []
    prev = None
    for lang in langs:
        val = (lang or "").strip()
        if not val or val == prev:
            continue
        out.append(val)
        prev = val
    return ",".join(out)
