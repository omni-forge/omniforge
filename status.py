#!/usr/bin/env python3
"""OmniForge project status dashboard.

Source: ZIP1 (primary). Shows document counts, token counts, checkpoint info,
        training progress, and disk usage.
"""

import os
import gzip
from pathlib import Path
from typing import Dict, Optional

import torch

import config
from train import latest_checkpoint


def file_size(path: Path) -> str:
    if not path.exists():
        return "N/A"
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 ** 3:
        return f"{size / 1024 ** 2:.1f} MB"
    return f"{size / 1024 ** 3:.1f} GB"


def count_lines_gz(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def count_tokens_bin(path: Path) -> int:
    if not path.exists():
        return 0
    return os.path.getsize(path) // 2  # uint16 = 2 bytes per token


def get_checkpoint_info() -> Optional[Dict]:
    ckpt = latest_checkpoint()
    if ckpt is None:
        return None
    try:
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        return {
            "path": ckpt.name,
            "step": data.get("step", "?"),
            "loss": data.get("loss", "?"),
        }
    except Exception:
        return {"path": ckpt.name, "step": "?", "loss": "?"}


def disk_usage(path: Path) -> str:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += os.path.getsize(f)
    if total < 1024:
        return f"{total} B"
    elif total < 1024 ** 2:
        return f"{total / 1024:.1f} KB"
    elif total < 1024 ** 3:
        return f"{total / 1024 ** 2:.1f} MB"
    return f"{total / 1024 ** 3:.1f} GB"


def main() -> None:
    print("=" * 60)
    print("  OmniForge Status Dashboard")
    print("=" * 60)

    # Dataset counts
    print("\n--- Dataset Documents ---")
    print(f"  Raw dataset:      {count_lines_gz(config.RAW_DATASET_PATH):>8,} docs")
    print(f"  Clean dataset:    {count_lines_gz(config.CLEAN_DATASET_PATH):>8,} docs")
    print(f"  Deduped dataset:  {count_lines_gz(config.DEDUPED_DATASET_PATH):>8,} docs")

    # Token counts
    print("\n--- Tokenized Splits ---")
    for split, path in [("Train", config.TRAIN_BIN_PATH),
                         ("Val", config.VAL_BIN_PATH),
                         ("Test", config.TEST_BIN_PATH)]:
        tokens = count_tokens_bin(path)
        print(f"  {split}: {tokens:>12,} tokens")

    # Checkpoint info
    print("\n--- Latest Checkpoint ---")
    ckpt_info = get_checkpoint_info()
    if ckpt_info:
        print(f"  File:  {ckpt_info['path']}")
        print(f"  Step:  {ckpt_info['step']}")
        print(f"  Loss:  {ckpt_info['loss']}")
    else:
        print("  No checkpoints found.")

    # Training progress
    if ckpt_info and ckpt_info["step"] != "?":
        progress = 100.0 * int(ckpt_info["step"]) / config.MAX_STEPS
        print(f"\n--- Training Progress ---")
        print(f"  {ckpt_info['step']:,} / {config.MAX_STEPS:,} steps ({progress:.1f}%)")
        bar_len = 40
        filled = int(bar_len * progress / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  [{bar}] {progress:.1f}%")

    # Disk usage
    print("\n--- Disk Usage ---")
    for d in config.DIRECTORIES:
        if d.exists():
            print(f"  {d.name}: {disk_usage(d):>10}")
    print()


if __name__ == "__main__":
    main()