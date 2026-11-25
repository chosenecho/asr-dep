"""Isolated per-GPU inference worker process (supports vLLM, Transformers, and Mock)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.alignment import has_alignable_text
from app.audio import read_chunks_from_wav

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
    aligner_batch_size: int = 6,
    prep_executor: Optional[ThreadPoolExecutor] = None,
    return_ts_flags: Optional[List[bool]] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Run Qwen3-ForcedAligner on alignable chunks using bounded sub-batches and
    CPU/GPU pipelining (parallel Whisper mel preprocessing overlapped with GPU forward),
    with per-chunk fallback so a single bad chunk can never crash a multi-hour job.
    """
    n = len(texts)
    per_chunk_words: List[List[Dict[str, Any]]] = [[] for _ in range(n)]

    aligner = getattr(model, "forced_aligner", None)
    if aligner is None:
        return per_chunk_words

    align_indices: List[int] = []
    align_audios: List[tuple] = []
    align_texts: List[str] = []
    align_langs: List[str] = []

    for i in range(n):
        if return_ts_flags is not None and not return_ts_flags[i]:
            continue
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

    def _convert_raw_items(raw_items: List[Dict[str, Any]], offset_sec: float) -> List[Dict[str, Any]]:
        return [
            {
                "text": str(it["text"]),
                "start": round(float(it["start_time"]) / 1000.0 + offset_sec, 3),
                "end": round(float(it["end_time"]) / 1000.0 + offset_sec, 3),
            }
            for it in raw_items
        ]

    def _fallback_single(
        sub_indices: List[int],
        sub_audios: List[tuple],
        sub_texts: List[str],
        sub_langs: List[str],
    ) -> None:
        for idx, aud, txt, lang in zip(sub_indices, sub_audios, sub_texts, sub_langs):
            try:
                single_res = aligner.align(audio=aud, text=txt, language=lang)[0]
                per_chunk_words[idx] = _convert(single_res, offsets[idx])
            except Exception as single_err:
                logger.warning(
                    "Forced alignment failed for chunk offset=%.2fs (%s); keeping ASR text only.",
                    offsets[idx],
                    single_err,
                )

    step = max(1, int(aligner_batch_size))
    sub_batches = [
        (
            align_indices[pos : pos + step],
            align_audios[pos : pos + step],
            align_texts[pos : pos + step],
            align_langs[pos : pos + step],
        )
        for pos in range(0, len(align_indices), step)
    ]

    can_pipeline = (
        prep_executor is not None
        and hasattr(aligner, "aligner_processor")
        and hasattr(aligner, "processor")
        and hasattr(aligner, "model")
        and hasattr(aligner, "timestamp_token_id")
        and hasattr(aligner, "timestamp_segment_time")
    )

    if can_pipeline:
        import torch

        def _prep_cpu(sub_audios_raw: List[Any], sub_input_texts: List[str]) -> Any:
            return aligner.processor(
                text=sub_input_texts,
                audio=sub_audios_raw,
                return_tensors="pt",
                padding=True,
            )

        # Encode timestamp text templates on main thread (<1ms), then dispatch CPU mel
        # extraction for ALL sub-batches concurrently so GPU never stalls between sub-batches.
        prepared_jobs = []
        for sub_indices, sub_audios, sub_texts, sub_langs in sub_batches:
            try:
                word_lists = []
                input_texts = []
                for t, lang in zip(sub_texts, sub_langs):
                    wl, inp_txt = aligner.aligner_processor.encode_timestamp(t, lang)
                    word_lists.append(wl)
                    input_texts.append(inp_txt)
                raw_wavs = [a[0] if isinstance(a, tuple) else a for a in sub_audios]
                fut = prep_executor.submit(_prep_cpu, raw_wavs, input_texts)
                prepared_jobs.append((sub_indices, sub_audios, sub_texts, sub_langs, word_lists, fut))
            except Exception as prep_err:
                logger.warning(
                    "Forced alignment template prep failed (%s); retrying sub-batch individually.",
                    prep_err,
                )
                _fallback_single(sub_indices, sub_audios, sub_texts, sub_langs)

        device = aligner.model.device
        dtype = aligner.model.dtype
        ts_token_id = aligner.timestamp_token_id
        ts_seg_time = aligner.timestamp_segment_time

        for sub_indices, sub_audios, sub_texts, sub_langs, word_lists, fut in prepared_jobs:
            inputs = None
            logits = None
            output_ids = None
            try:
                inputs = fut.result()
                with torch.inference_mode():
                    gpu_inputs = inputs.to(device, non_blocking=True).to(dtype)
                    logits = aligner.model.thinker(**gpu_inputs).logits
                    output_ids = logits.argmax(dim=-1)
                    input_ids_gpu = gpu_inputs["input_ids"]
                    masked_cpu = [
                        (out_id[in_id == ts_token_id] * ts_seg_time).to("cpu").numpy()
                        for in_id, out_id in zip(input_ids_gpu, output_ids)
                    ]
                    del gpu_inputs, logits, output_ids
                    logits = None
                    output_ids = None
                for idx, wl, ts_ms in zip(sub_indices, word_lists, masked_cpu):
                    parsed = aligner.aligner_processor.parse_timestamp(wl, ts_ms)
                    per_chunk_words[idx] = _convert_raw_items(parsed, offsets[idx])
            except Exception as batch_err:
                logger.warning(
                    "Pipelined forced alignment sub-batch failed (%s); retrying chunks individually.",
                    batch_err,
                )
                try:
                    del inputs, logits, output_ids
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                _fallback_single(sub_indices, sub_audios, sub_texts, sub_langs)
        return per_chunk_words

    for sub_indices, sub_audios, sub_texts, sub_langs in sub_batches:
        try:
            aligned_batch = aligner.align(
                audio=sub_audios,
                text=sub_texts,
                language=sub_langs,
            )
            for idx, aligned_res in zip(sub_indices, aligned_batch):
                per_chunk_words[idx] = _convert(aligned_res, offsets[idx])
        except Exception as batch_err:
            logger.warning(
                "Batch forced alignment failed (%s); retrying chunks individually.", batch_err
            )
            _fallback_single(sub_indices, sub_audios, sub_texts, sub_langs)

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


def _process_real_superbatch(
    model: Any,
    loaded_items: List[Tuple[Dict[str, Any], List[tuple]]],
    logger: logging.Logger,
    aligner_batch_size: int,
    prep_executor: ThreadPoolExecutor,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Run ONE coalesced vLLM ASR pass across chunks from multiple concurrent jobs,
    followed by pipelined sub-batch ForcedAligner, then split results per batch_id.
    """
    all_audios: List[tuple] = []
    all_contexts: List[str] = []
    all_req_langs: List[Optional[str]] = []
    all_offsets: List[float] = []
    all_return_ts: List[bool] = []
    slices: List[Tuple[Dict[str, Any], int, int]] = []

    for work_item, item_audios in loaded_items:
        chunks = work_item["chunks"]
        req_lang = normalize_requested_language(work_item.get("language"))
        context = work_item.get("context") or ""
        return_ts = bool(work_item.get("return_timestamps", True))
        start_idx = len(all_audios)
        for ch, aud in zip(chunks, item_audios):
            all_audios.append(aud)
            all_contexts.append(context)
            all_req_langs.append(req_lang)
            all_offsets.append(float(ch["start_sec"]))
            all_return_ts.append(return_ts)
        slices.append((work_item, start_idx, len(all_audios)))

    if not all_audios:
        return [(wi["batch_id"], []) for wi, _, _ in slices]

    asr_outputs = model.transcribe(
        audio=all_audios,
        context=all_contexts,
        language=all_req_langs,
        return_time_stamps=False,
    )

    texts = [r.text or "" for r in asr_outputs]
    langs = [
        (r.language or (req_l or "")) for r, req_l in zip(asr_outputs, all_req_langs)
    ]

    if any(all_return_ts):
        per_chunk_words = _align_chunks_safely(
            model=model,
            audios=all_audios,
            texts=texts,
            langs=langs,
            offsets=all_offsets,
            logger=logger,
            aligner_batch_size=aligner_batch_size,
            prep_executor=prep_executor,
            return_ts_flags=all_return_ts,
        )
    else:
        per_chunk_words = [[] for _ in all_audios]

    outputs: List[Tuple[str, List[Dict[str, Any]]]] = []
    for work_item, s_idx, e_idx in slices:
        chunks = work_item["chunks"]
        item_results: List[Dict[str, Any]] = []
        for ch, lang, txt, words in zip(
            chunks,
            langs[s_idx:e_idx],
            texts[s_idx:e_idx],
            per_chunk_words[s_idx:e_idx],
        ):
            item_results.append(
                {
                    "chunk_index": int(ch["chunk_index"]),
                    "start_sec": float(ch["start_sec"]),
                    "end_sec": float(ch["end_sec"]),
                    "language": lang,
                    "text": txt,
                    "words": words,
                }
            )
        outputs.append((work_item["batch_id"], item_results))
    return outputs


def _prefetch_loop(
    work_queue: Any,
    prefetch_queue: "queue.Queue[Optional[Tuple[List[Tuple[Dict[str, Any], List[tuple]]], List[Tuple[str, str]]]]]",
    backend: str,
    max_batch: int,
    num_workers: int,
    coalesce_wait_sec: float,
    logger: logging.Logger,
) -> None:
    """
    Background producer inside each GPU worker:
    1. Coalesces queued mini-batches across concurrent jobs up to `max_batch` chunks.
    2. Preloads WAV chunks from disk into RAM in parallel WHILE GPU runs the previous batch.
    """
    overflow_item: Optional[Dict[str, Any]] = None
    shutdown_pending = False
    io_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wav-prefetch")

    def _load_item(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[List[tuple]], Optional[str]]:
        if backend == "mock":
            return item, [], None
        try:
            audios = read_chunks_from_wav(Path(item["wav_path"]), item["chunks"])
            return item, audios, None
        except Exception as exc:
            return item, None, str(exc)

    try:
        while True:
            if overflow_item is not None:
                first = overflow_item
                overflow_item = None
            elif shutdown_pending:
                prefetch_queue.put(None)
                return
            else:
                first = work_queue.get()
                if first is None:
                    prefetch_queue.put(None)
                    return

            items: List[Dict[str, Any]] = [first]
            total_chunks = len(first.get("chunks") or [])
            deadline = time.monotonic() + coalesce_wait_sec

            while total_chunks < max_batch:
                try:
                    if num_workers > 1:
                        nxt = work_queue.get_nowait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining > 0.001:
                            nxt = work_queue.get(timeout=remaining)
                        else:
                            nxt = work_queue.get_nowait()
                except queue.Empty:
                    break

                if nxt is None:
                    shutdown_pending = True
                    break

                nxt_len = len(nxt.get("chunks") or [])
                if total_chunks + nxt_len <= max_batch:
                    items.append(nxt)
                    total_chunks += nxt_len
                else:
                    overflow_item = nxt
                    break

            loaded_items: List[Tuple[Dict[str, Any], List[tuple]]] = []
            failed_items: List[Tuple[str, str]] = []

            if len(items) == 1:
                loaded_results = [_load_item(items[0])]
            else:
                loaded_results = list(io_pool.map(_load_item, items))

            for item, audios, err in loaded_results:
                if err is not None or audios is None:
                    logger.warning("Failed to preload WAV for batch %s: %s", item.get("batch_id"), err)
                    failed_items.append((item["batch_id"], err or "WAV preload failed"))
                else:
                    loaded_items.append((item, audios))

            prefetch_queue.put((loaded_items, failed_items))
    finally:
        io_pool.shutdown(wait=False)


def gpu_worker_main(
    worker_cfg: Dict[str, Any],
    work_queue: Any,
    result_queue: Any,
    ready_queue: Any,
) -> None:
    """Entrypoint for each spawned GPU worker process."""
    gpu_id = str(worker_cfg["gpu_id"])
    backend = str(worker_cfg["asr_backend"])
    max_batch = max(1, int(worker_cfg.get("max_inference_batch_size", 16)))
    aligner_batch_size = max(1, int(worker_cfg.get("aligner_batch_size", 6)))
    num_workers = max(1, int(worker_cfg.get("num_workers", 1)))
    coalesce_wait_sec = max(0.0, int(worker_cfg.get("batch_coalesce_wait_ms", 15)) / 1000.0)

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

    prefetch_queue: "queue.Queue[Optional[Tuple[List[Tuple[Dict[str, Any], List[tuple]]], List[Tuple[str, str]]]]]" = queue.Queue(
        maxsize=1
    )
    prefetch_thread = threading.Thread(
        target=_prefetch_loop,
        args=(
            work_queue,
            prefetch_queue,
            backend,
            max_batch,
            num_workers,
            coalesce_wait_sec,
            logger,
        ),
        name=f"gpu-{gpu_id}-prefetcher",
        daemon=True,
    )
    prefetch_thread.start()

    align_prep_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix=f"gpu-{gpu_id}-alignprep")
    try:
        while True:
            bundle = prefetch_queue.get()
            if bundle is None:
                logger.info("Received shutdown signal; exiting worker on GPU %s.", gpu_id)
                break

            loaded_items, failed_items = bundle
            for failed_batch_id, err_msg in failed_items:
                result_queue.put(
                    {
                        "batch_id": failed_batch_id,
                        "gpu_id": gpu_id,
                        "ok": False,
                        "error": err_msg,
                    }
                )

            if not loaded_items:
                continue

            if backend == "mock":
                for work_item, _ in loaded_items:
                    batch_id = work_item["batch_id"]
                    try:
                        chunk_results = _process_mock_batch(work_item)
                        result_queue.put(
                            {
                                "batch_id": batch_id,
                                "gpu_id": gpu_id,
                                "ok": True,
                                "chunks": chunk_results,
                            }
                        )
                    except Exception as exc:
                        logger.exception("Mock batch %s failed on GPU %s", batch_id, gpu_id)
                        result_queue.put(
                            {
                                "batch_id": batch_id,
                                "gpu_id": gpu_id,
                                "ok": False,
                                "error": str(exc),
                            }
                        )
                continue

            try:
                per_batch_outputs = _process_real_superbatch(
                    model=model,
                    loaded_items=loaded_items,
                    logger=logger,
                    aligner_batch_size=aligner_batch_size,
                    prep_executor=align_prep_pool,
                )
                for batch_id, chunk_results in per_batch_outputs:
                    result_queue.put(
                        {
                            "batch_id": batch_id,
                            "gpu_id": gpu_id,
                            "ok": True,
                            "chunks": chunk_results,
                        }
                    )
            except Exception as exc:
                if len(loaded_items) == 1:
                    batch_id = loaded_items[0][0]["batch_id"]
                    logger.exception("Batch %s failed on GPU %s", batch_id, gpu_id)
                    result_queue.put(
                        {
                            "batch_id": batch_id,
                            "gpu_id": gpu_id,
                            "ok": False,
                            "error": str(exc),
                        }
                    )
                else:
                    logger.warning(
                        "Coalesced super-batch (%d items) failed on GPU %s (%s); retrying items individually.",
                        len(loaded_items),
                        gpu_id,
                        exc,
                    )
                    for single_pair in loaded_items:
                        batch_id = single_pair[0]["batch_id"]
                        try:
                            out = _process_real_superbatch(
                                model=model,
                                loaded_items=[single_pair],
                                logger=logger,
                                aligner_batch_size=aligner_batch_size,
                                prep_executor=align_prep_pool,
                            )
                            for b_id, chunk_results in out:
                                result_queue.put(
                                    {
                                        "batch_id": b_id,
                                        "gpu_id": gpu_id,
                                        "ok": True,
                                        "chunks": chunk_results,
                                    }
                                )
                        except Exception as single_exc:
                            logger.exception("Batch %s failed on GPU %s", batch_id, gpu_id)
                            result_queue.put(
                                {
                                    "batch_id": batch_id,
                                    "gpu_id": gpu_id,
                                    "ok": False,
                                    "error": str(single_exc),
                                }
                            )
    finally:
        align_prep_pool.shutdown(wait=False)
        prefetch_thread.join(timeout=3.0)
