#!/usr/bin/env python3
"""Example client for calling the Qwen3-ASR FastAPI service (async long audio & sync)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def submit_and_wait_long_audio(
    base_url: str,
    api_key: str,
    audio_path: Path,
    language: str | None,
    context: str,
    save_srt: Path | None,
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    with httpx.Client(base_url=base_url, headers=headers, timeout=120.0) as client:
        print(f"Uploading {audio_path} to /v1/asr/tasks ...")
        with audio_path.open("rb") as f:
            data = {
                "context": context,
                "return_timestamps": "true",
                "include_words": "true",
            }
            if language:
                data["language"] = language
            resp = client.post(
                "/v1/asr/tasks",
                files={"file": (audio_path.name, f)},
                data=data,
            )
            resp.raise_for_status()
            task = resp.json()

        task_id = task["task_id"]
        print(f"Task accepted: task_id={task_id}")

        while True:
            poll = client.get(f"/v1/asr/tasks/{task_id}")
            poll.raise_for_status()
            info = poll.json()
            pct = info["progress"] * 100
            print(
                f"  status={info['status']} progress={pct:.1f}% "
                f"chunks={info['completed_chunks']}/{info['total_chunks']}"
            )
            if info["status"] == "completed":
                result = info["result"]
                print("\n=== Transcription Completed ===")
                print(f"Language: {result['language']}")
                print(
                    f"Duration: {result['duration_sec']}s | "
                    f"Elapsed: {result['processing_time_sec']}s | RTF: {result['rtf']}"
                )
                print(f"Full Text:\n{result['text']}\n")
                print(f"Segments ({len(result['segments'])}):")
                for seg in result["segments"][:10]:
                    print(f"  [{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
                if len(result["segments"]) > 10:
                    print(f"  ... ({len(result['segments']) - 10} more segments)")

                if save_srt:
                    srt_resp = client.get(
                        f"/v1/asr/tasks/{task_id}/subtitle", params={"format": "srt"}
                    )
                    srt_resp.raise_for_status()
                    save_srt.write_text(srt_resp.text, encoding="utf-8")
                    print(f"SRT saved to {save_srt}")
                break

            if info["status"] == "failed":
                raise RuntimeError(f"Task failed: {info.get('error')}")

            time.sleep(1.5)


def call_sync(
    base_url: str,
    api_key: str,
    audio_path: Path,
    language: str | None,
    context: str,
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(base_url=base_url, headers=headers, timeout=600.0) as client:
        with audio_path.open("rb") as f:
            data = {
                "context": context,
                "return_timestamps": "true",
                "include_words": "true",
                "response_format": "json",
            }
            if language:
                data["language"] = language
            resp = client.post(
                "/v1/asr/transcribe",
                files={"file": (audio_path.name, f)},
                data=data,
            )
            resp.raise_for_status()
            print(json.dumps(resp.json(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test client for Qwen3-ASR FastAPI service.")
    parser.add_argument("audio", type=Path, help="Path to local audio/video file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--language", default=None, help="e.g. Chinese / English / zh / en")
    parser.add_argument("--context", default="", help="Optional hotwords / context prompt")
    parser.add_argument("--mode", choices=["async", "sync"], default="async")
    parser.add_argument("--save-srt", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "async":
        submit_and_wait_long_audio(
            args.base_url, args.api_key, args.audio, args.language, args.context, args.save_srt
        )
    else:
        call_sync(args.base_url, args.api_key, args.audio, args.language, args.context)


if __name__ == "__main__":
    main()
