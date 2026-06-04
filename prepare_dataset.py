#!/usr/bin/env python3
"""Re-tokenizes deduped dataset using TinyLlama tokenizer."""

import gzip, json, random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_NAME   = "deepseek-ai/deepseek-coder-1.3b-base"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_PATH   = DATA_DIR / "clean" / "deduped_dataset.jsonl.gz"
OUT_DIR      = DATA_DIR / "tokenized"
TRAIN_BIN    = OUT_DIR / "train.bin"
VAL_BIN      = OUT_DIR / "val.bin"
TEST_BIN     = OUT_DIR / "test.bin"

CONTEXT_LENGTH = 2048
DTYPE          = np.uint16
SEED           = 1337
TRAIN_SPLIT    = 0.95
VAL_SPLIT      = 0.04

def choose_split(rng):
    x = rng.random()
    if x < TRAIN_SPLIT: return "train"
    if x < TRAIN_SPLIT + VAL_SPLIT: return "val"
    return "test"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] Loading tokenizer from {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    print(f"[prepare] EOS id: {eos_id}  PAD id: {pad_id}")

    handles = {
        "train": open(TRAIN_BIN, "wb"),
        "val":   open(VAL_BIN,   "wb"),
        "test":  open(TEST_BIN,  "wb"),
    }
    token_counts = {"train": 0, "val": 0, "test": 0}
    chunk_counts = {"train": 0, "val": 0, "test": 0}
    rng = random.Random(SEED)
    current, current_split = [], None
    docs = 0

    print(f"[prepare] Reading {INPUT_PATH} ...")
    with gzip.open(INPUT_PATH, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line: continue
            try: record = json.loads(line)
            except json.JSONDecodeError: continue
            text = record.get("content", "")
            if not text: continue
            docs += 1
            split = choose_split(rng)
            tokens = tokenizer.encode(text) + [eos_id]

            if current_split and current_split != split and current:
                while len(current) < CONTEXT_LENGTH: current.append(pad_id)
                handles[current_split].write(np.asarray(current, dtype=DTYPE).tobytes())
                token_counts[current_split] += CONTEXT_LENGTH
                chunk_counts[current_split] += 1
                current, current_split = [], None

            if current_split is None: current_split = split

            for tok in tokens:
                current.append(tok)
                if len(current) == CONTEXT_LENGTH:
                    handles[current_split].write(np.asarray(current, dtype=DTYPE).tobytes())
                    token_counts[current_split] += CONTEXT_LENGTH
                    chunk_counts[current_split] += 1
                    current = []

    if current:
        while len(current) < CONTEXT_LENGTH: current.append(pad_id)
        handles[current_split].write(np.asarray(current, dtype=DTYPE).tobytes())
        token_counts[current_split] += CONTEXT_LENGTH
        chunk_counts[current_split] += 1

    for h in handles.values(): h.close()

    print(f"\n[prepare] Done. {docs:,} documents processed.")
    for s in ["train", "val", "test"]:
        print(f"[prepare]   {s}: {token_counts[s]:,} tokens ({chunk_counts[s]:,} chunks)")

if __name__ == "__main__":
    main()
