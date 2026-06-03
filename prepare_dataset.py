#!/usr/bin/env python3
"""Tokenize, pack, and split OmniForge training data into binary memmap files.

Source: ZIP1 (primary). Uses greedy packing, memory-mapped numpy arrays,
        proper train/val/test splitting.
"""

import gzip
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

import config


DTYPE = np.uint16


def choose_split(rng: random.Random) -> str:
    x = rng.random()
    train, val, _test = config.SPLIT_RATIOS
    if x < train:
        return "train"
    if x < train + val:
        return "val"
    return "test"


def append_chunk(split: str, chunk: List[int],
                 handles: Dict[str, object], token_counts: Dict[str, int],
                 chunk_counts: Dict[str, int]) -> None:
    if len(chunk) != config.CONTEXT_LENGTH:
        raise ValueError(f"Chunk length must be {config.CONTEXT_LENGTH}, got {len(chunk)}")
    array = np.asarray(chunk, dtype=DTYPE)
    handles[split].write(array.tobytes(order="C"))
    token_counts[split] += len(chunk)
    chunk_counts[split] += 1


def main() -> None:
    config.ensure_directories()

    print("[prepare] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(config.TOKENIZER_OUTPUT_DIR), use_fast=True)
    eod_id = tokenizer.convert_tokens_to_ids(config.EOD_TOKEN)
    if eod_id is None or eod_id < 0:
        raise RuntimeError("EOD token is missing from tokenizer. Train tokenizer first.")

    if config.VOCAB_SIZE > np.iinfo(DTYPE).max + 1:
        raise RuntimeError("Vocabulary too large for uint16 storage.")

    output_paths = {
        "train": config.TRAIN_BIN_PATH,
        "val": config.VAL_BIN_PATH,
        "test": config.TEST_BIN_PATH,
    }

    # Open binary output files
    handles = {}
    for split, path in output_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        handles[split] = open(path, "wb")

    token_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    chunk_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    rng = random.Random(config.SEED)

    # Current packing buffer
    current_chunk: List[int] = []
    current_chunk_split = None
    docs_processed = 0

    print("[prepare] Tokenizing and packing dataset...")
    with gzip.open(config.DEDUPED_DATASET_PATH, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("content", "")
            if not text:
                continue

            docs_processed += 1

            # Choose split for this document
            split = choose_split(rng)

            # Tokenize document
            tokens = tokenizer.encode(text)
            # Append EOD token to mark document boundary
            tokens.append(eod_id)

            # If we have a pending chunk from a different split, flush it
            if current_chunk_split is not None and current_chunk_split != split and current_chunk:
                # Pad and flush the old chunk
                while len(current_chunk) < config.CONTEXT_LENGTH:
                    current_chunk.append(config.PAD_TOKEN_ID)
                append_chunk(current_chunk_split, current_chunk, handles,
                             token_counts, chunk_counts)
                current_chunk = []
                current_chunk_split = None

            if current_chunk_split is None:
                current_chunk_split = split

            # Greedy packing: fill current chunk
            for token in tokens:
                current_chunk.append(token)
                if len(current_chunk) == config.CONTEXT_LENGTH:
                    append_chunk(current_chunk_split, current_chunk, handles,
                                 token_counts, chunk_counts)
                    current_chunk = []

    # Flush remaining tokens
    if current_chunk:
        while len(current_chunk) < config.CONTEXT_LENGTH:
            current_chunk.append(config.PAD_TOKEN_ID)
        append_chunk(current_chunk_split, current_chunk, handles,
                     token_counts, chunk_counts)

    # Close all files
    for h in handles.values():
        h.close()

    total_tokens = sum(token_counts.values())
    print(f"\n[prepare] Done. Processed {docs_processed:,} documents.")
    print(f"[prepare] Total tokens written: {total_tokens:,}")
    for split in ["train", "val", "test"]:
        print(f"[prepare]   {split}: {token_counts[split]:,} tokens "
              f"({chunk_counts[split]:,} chunks)")
    print(f"[prepare] Dataset files saved to {config.TOKENIZED_DATA_DIR}")


if __name__ == "__main__":
    main()