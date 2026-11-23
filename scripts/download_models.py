#!/usr/bin/env python3
"""Download Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B weights to local ./models directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = [
    ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B"),
    ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner-0.6B"),
]


def download_via_modelscope(repo_id: str, local_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "modelscope.cli.cli",
        "download",
        "--model",
        repo_id,
        "--local_dir",
        str(local_dir),
    ]
    print(f"[ModelScope] Downloading {repo_id} -> {local_dir}")
    subprocess.run(cmd, check=True)


def download_via_hf(repo_id: str, local_dir: Path) -> None:
    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        "--local-dir",
        str(local_dir),
    ]
    print(f"[HuggingFace] Downloading {repo_id} -> {local_dir}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3-ASR weights locally.")
    parser.add_argument(
        "--source",
        choices=["modelscope", "hf"],
        default="modelscope",
        help="Download source (default: modelscope, faster in Mainland China).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models"),
        help="Target directory to store downloaded model folders.",
    )
    parser.add_argument(
        "--skip-aligner",
        action="store_true",
        help="Skip downloading Qwen3-ForcedAligner-0.6B (only if timestamps are not needed).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, folder_name in MODELS:
        if args.skip_aligner and "ForcedAligner" in repo_id:
            continue
        target = args.output_dir / folder_name
        if args.source == "modelscope":
            download_via_modelscope(repo_id, target)
        else:
            download_via_hf(repo_id, target)

    print("All requested models downloaded successfully.")


if __name__ == "__main__":
    main()
