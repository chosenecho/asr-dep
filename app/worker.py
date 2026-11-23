"""Isolated per-GPU inference worker process (supports vLLM, Transformers, and Mock)."""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.alignment import has_alignable_text
from app.audio import read_chunk_waveform

# Qwen3-ForcedAligner-0.6B officially supports these 11 languages.
ALIGNER_SUPPORTED_LANGUAGES = {
    "chinese",
    "english",
    "cantonese",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
}

# Map common ISO-639-1 / shorthand codes to Qwen3-ASR canonical language names.
ISO_TO_QWEN_LANGUAGE: Dict[str, str] = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "cmn": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def normalize_requested_language(lang: Optional[str]) -> Optional[str]:
    """Convert ISO code (e.g. 'zh', 'en') or mixed-case name to Qwen3-ASR canonical name."""
    if not lang or not lang.strip():
        return None
    raw = lang.strip()
    low = raw.lower()
    if low in ("auto", "none"):
        return None
    if low in ISO_TO_QWEN_LANGUAGE:
        return ISO_TO_QWEN_LANGUAGE[low]
    return raw[:1].upper() + raw[1:].lower()


def _load_real_model(worker_cfg: Dict[str, Any], logger: logging.Logger) -> Any:
    import torch
    from qwen_asr import Qwen3ASRModel

    backend = worker_cfg["asr_backend"]
    asr_model_path = worker_cfg["asr_model_path"]
    aligner_path = (worker_cfg.get("aligner_model_path") or "").strip() or None
    use_flash = bool(worker_cfg.get("use_flash_attention", False))

    aligner_kwargs: Optional[Dict[str, Any]] = None
    if aligner_path:
        aligner_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "cuda:0",
        }
        if use_flash:
            aligner_kwargs["attn_implementation"] = "flash_attention_2"

    logger.info(
        "Loading Qwen3-ASR backend=%s model=%s aligner=%s on visible cuda:0 (physical GPU %s)",
        backend,
        asr_model_path,
        aligner_path,
        worker_cfg["gpu_id"],
    )

    if backend == "vllm":
        return Qwen3ASRModel.LLM(
            model=asr_model_path,
            gpu_memory_utilization=float(worker_cfg["gpu_memory_utilization"]),
            max_inference_batch_size=int(worker_cfg["max_inference_batch_size"]),
            max_new_tokens=int(worker_cfg["max_new_tokens"]),
            forced_aligner=aligner_path,
            forced_aligner_kwargs=aligner_kwargs,
        )

    if backend == "transformers":
        tf_kwargs: Dict[str, Any] = {
            "dtype": torch.bfloat16,
            "device_map": "cuda:0",
            "max_inference_batch_size": int(worker_cfg["max_inference_batch_size"]),
            "max_new_tokens": int(worker_cfg["max_new_tokens"]),
            "forced_aligner": aligner_path,
            "forced_aligner_kwargs": aligner_kwargs,
        }
        if use_flash:
            tf_kwargs["attn_implementation"] = "flash_attention_2"
        return Qwen3ASRModel.from_pretrained(asr_model_path, **tf_kwargs)

    raise ValueError(f"Unsupported backend: {backend}")


def _align_chunks_safely(
    model: Any,
    audios: List[tuple],
    texts: List[str],
    langs: List[str],
    offsets: List[float],
    logger: logging.Logger,
) -> List[List[Dict[str, Any]]]:
    """
    Run Qwen3-ForcedAligner on alignable chunks with per-chunk fallback so a single
    silent/unsupported chunk can never crash a multi-hour transcription job.
    """
    n = len(texts)
    per_chunk_words: List[List[Dict[str, Any]]] = [[] for _ in range(n)]

    if getattr(model, "forced_aligner", None) is None:
        return per_chunk_words

    align_indices: List[int] = []
    align_audios: List[tuple] = []
    align_texts: List[str] = []
    align_langs: List[str] = []

    for i in range(n):
        txt = (texts[i] or "").strip()
        lang = (langs[i] or "").strip()
        if not txt or not has_alignable_text(txt):
            continue
        if lang.lower() not in ALIGNER_SUPPORTED_LANGUAGES:
            logger.debug(
                "Skipping forced alignment for chunk %d (language=%r not in aligner 11-lang set).",
                i,
                lang,
            )
            continue
        align_indices.append(i)
        align_audios.append(audios[i])
        align_texts.append(txt)
        align_langs.append(lang)

    if not align_indices:
        return per_chunk_words

    def _convert(aligned_res: Any, offset_sec: float) -> List[Dict[str, Any]]:
        if aligned_res is None:
            return []
        return [
            {
                "text": str(item.text),
                "start": round(float(item.start_time) + offset_sec, 3),
                "end": round(float(item.end_time) + offset_sec, 3),
            }
            for item in aligned_res.items
        ]

    try:
        aligned_batch = model.forced_aligner.align(
            audio=align_audios,
            text=align_texts,
            language=align_langs,
        )
        for idx, aligned_res in zip(align_indices, aligned_batch):
            per_chunk_words[idx] = _convert(aligned_res, offsets[idx])
    except Exception as batch_err:
        logger.warning(
            "Batch forced alignment failed (%s); retrying chunks individually.", batch_err
        )
        for idx, aud, txt, lang in zip(
            align_indices, align_audios, align_texts, align_langs
        ):
            try:
                single_res = model.forced_aligner.align(
                    audio=aud, text=txt, language=lang
                )[0]
                per_chunk_words[idx] = _convert(single_res, offsets[idx])
            except Exception as single_err:
                logger.warning(
                    "Forced alignment failed for chunk offset=%.2fs (%s); keeping ASR text only.",
                    offsets[idx],
                    single_err,
                )

    return per_chunk_words


def _process_mock_batch(work_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic mock inference for local CPU tests without GPU hardware."""
    req_lang = normalize_requested_language(work_item.get("language")) or "Chinese"
    return_ts = bool(work_item.get("return_timestamps", True))
    results: List[Dict[str, Any]] = []

    for ch in work_item["chunks"]:
        c_idx = int(ch["chunk_index"])
        start_sec = float(ch["start_sec"])
        end_sec = float(ch["end_sec"])
        dur = max(0.2, end_sec - start_sec)

        if req_lang == "English":
            tokens = ["Hello", "world", f"chunk{c_idx}", "done"]
            text = f"Hello world. chunk{c_idx} done."
        else:
            tokens = ["你", "好", "世", "界", "第", str(c_idx + 1), "段", "完", "成"]
            text = f"你好世界。第{c_idx + 1}段完成。"

        words: List[Dict[str, Any]] = []
        if return_ts:
            step = dur / len(tokens)
            for ti, tok in enumerate(tokens):
                w_start = round(start_sec + ti * step, 3)
                w_end = round(min(end_sec, start_sec + (ti + 0.9) * step), 3)
                words.append({"text": tok, "start": w_start, "end": w_end})

        results.append(
            {
                "chunk_index": c_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "language": req_lang,
                "text": text,
                "words": words,
            }
        )
    return results


def _process_real_batch(
    model: Any, work_item: Dict[str, Any], logger: logging.Logger
) -> List[Dict[str, Any]]:
    wav_path = Path(work_item["wav_path"])
    chunks = work_item["chunks"]
    req_lang = normalize_requested_language(work_item.get("language"))
    context = work_item.get("context") or ""
    return_ts = bool(work_item.get("return_timestamps", True))

    audios = [
        read_chunk_waveform(wav_path, int(ch["start_frame"]), int(ch["end_frame"]))
        for ch in chunks
    ]
    offsets = [float(ch["start_sec"]) for ch in chunks]

    asr_outputs = model.transcribe(
        audio=audios,
        context=context,
        language=req_lang,
        return_time_stamps=False,
    )

    texts = [r.text or "" for r in asr_outputs]
    langs = [r.language or (req_lang or "") for r in asr_outputs]

    if return_ts:
        per_chunk_words = _align_chunks_safely(
            model=model,
            audios=audios,
            texts=texts,
            langs=langs,
            offsets=offsets,
            logger=logger,
        )
    else:
        per_chunk_words = [[] for _ in chunks]

    results: List[Dict[str, Any]] = []
    for ch, lang, txt, words in zip(chunks, langs, texts, per_chunk_words):
        results.append(
            {
                "chunk_index": int(ch["chunk_index"]),
                "start_sec": float(ch["start_sec"]),
                "end_sec": float(ch["end_sec"]),
                "language": lang,
                "text": txt,
                "words": words,
            }
        )
    return results


def gpu_worker_main(
    worker_cfg: Dict[str, Any],
    work_queue: Any,
    result_queue: Any,
    ready_queue: Any,
) -> None:
    """Entrypoint for each spawned GPU worker process."""
    gpu_id = str(worker_cfg["gpu_id"])
    backend = str(worker_cfg["asr_backend"])

    if backend != "mock":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    logging.basicConfig(
        level=getattr(logging, str(worker_cfg.get("log_level", "INFO")).upper(), logging.INFO),
        format=f"%(asctime)s [%(levelname)s] [gpu-worker-{gpu_id}] %(message)s",
    )
    logger = logging.getLogger(f"asr.worker.{gpu_id}")

    model = None
    try:
        if backend != "mock":
            model = _load_real_model(worker_cfg, logger)
        else:
            logger.info("Mock worker initialized for GPU slot %s.", gpu_id)
        ready_queue.put({"gpu_id": gpu_id, "status": "ready", "pid": os.getpid()})
    except Exception as exc:
        logger.exception("Failed to initialize worker on GPU %s", gpu_id)
        ready_queue.put(
            {
                "gpu_id": gpu_id,
                "status": "error",
                "error": f"{exc}\n{traceback.format_exc()}",
            }
        )
        return

    while True:
        work_item = work_queue.get()
        if work_item is None:
            logger.info("Received shutdown signal; exiting worker on GPU %s.", gpu_id)
            break

        batch_id = work_item["batch_id"]
        try:
            if backend == "mock":
                chunk_results = _process_mock_batch(work_item)
            else:
                chunk_results = _process_real_batch(model, work_item, logger)
            result_queue.put(
                {
                    "batch_id": batch_id,
                    "gpu_id": gpu_id,
                    "ok": True,
                    "chunks": chunk_results,
                }
            )
        except Exception as exc:
            logger.exception("Batch %s failed on GPU %s", batch_id, gpu_id)
            result_queue.put(
                {
                    "batch_id": batch_id,
                    "gpu_id": gpu_id,
                    "ok": False,
                    "error": str(exc),
                }
            )
