#!/usr/bin/env python3
"""Train a BPE tokenizer for OmniForge.

Trains a HuggingFace BPE tokenizer with byte-level pre-tokenization
and all required special tokens. Saves in HuggingFace-compatible format
so it can be loaded with AutoTokenizer.from_pretrained().
"""

import gzip
import json
import os
import sys
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

import config


# Five hardcoded snippets for post-training verification
VERIFICATION_SNIPPETS = [
    "def add(a, b):\n    return a + b\n",
    "class Cache:\n    def __init__(self):\n        self.items = {}\n",
    "import os\nfrom pathlib import Path\nprint(Path.cwd())\n",
    "async def fetch(session, url):\n    async with session.get(url) as response:\n        return await response.text()\n",
    "function greet(name) {\n  return `Hello, ${name}`;\n}\n",
]


def document_iterator(max_docs: int = config.MAX_TOKENIZER_DOCS) -> Iterator[str]:
    """Yield text documents from the deduplicated dataset."""
    count = 0
    with gzip.open(config.DEDUPED_DATASET_PATH, "rt", encoding="utf-8", errors="replace") as reader:
        for line in reader:
            if count >= max_docs:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(record.get("content", ""))
            if text.strip():
                count += 1
                yield text
    print(f"[tokenizer] Iterated {count:,} documents for training.")


def build_tokenizer() -> Tokenizer:
    """Construct a bare BPE tokenizer with byte-level pre-tokenization."""
    tokenizer = Tokenizer(models.BPE(unk_token=config.UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def main() -> None:
    if not config.DEDUPED_DATASET_PATH.exists():
        print(f"[tokenizer] ERROR: Deduped dataset not found at {config.DEDUPED_DATASET_PATH}")
        print("[tokenizer] Run deduplicator.py first.")
        sys.exit(1)

    config.ensure_directories()
    config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=config.TOKENIZER_VOCAB_SIZE,
        min_frequency=config.TOKENIZER_MIN_FREQUENCY,
        special_tokens=config.SPECIAL_TOKENS,
        show_progress=True,
    )

    print(f"[tokenizer] Training BPE tokenizer (vocab_size={config.TOKENIZER_VOCAB_SIZE:,})...")
    tokenizer.train_from_iterator(
        document_iterator(),
        trainer=trainer,
        length=config.MAX_TOKENIZER_DOCS,
    )

    # Save raw tokenizer JSON
    tokenizer_json_path = config.TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    # Wrap in HuggingFace PreTrainedTokenizerFast and save
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        unk_token=config.UNK_TOKEN,
        pad_token=config.PAD_TOKEN,
        bos_token=config.BOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        additional_special_tokens=[config.EOD_TOKEN],
        model_max_length=config.CONTEXT_LENGTH,
        padding_side="right",
    )
    hf_tokenizer.save_pretrained(str(config.TOKENIZER_DIR))
    print(f"[tokenizer] Saved HuggingFace-compatible tokenizer to {config.TOKENIZER_DIR}")
    print(f"[tokenizer] Vocabulary size: {hf_tokenizer.vocab_size:,}")

    # Verify special token IDs match config
    for token, expected_id in config.SPECIAL_TOKEN_IDS.items():
        actual_id = hf_tokenizer.convert_tokens_to_ids(token)
        status = "OK" if actual_id == expected_id else f"MISMATCH (got {actual_id}, expected {expected_id})"
        print(f"[tokenizer] Special token {token!r}: id={actual_id} [{status}]")

    # Encode/decode verification
    print("\n[tokenizer] Verification — encoding and decoding 5 code snippets:")
    for index, snippet in enumerate(VERIFICATION_SNIPPETS, start=1):
        encoded = hf_tokenizer.encode(snippet, add_special_tokens=False)
        decoded = hf_tokenizer.decode(encoded, skip_special_tokens=False)
        match = "✓" if decoded == snippet else "✗"
        print(f"  [{index}] {len(snippet)} chars → {len(encoded)} tokens → roundtrip {match}")
        print(f"       input:   {snippet[:60].strip()!r}")
        print(f"       decoded: {decoded[:60].strip()!r}")


if __name__ == "__main__":
    main()
