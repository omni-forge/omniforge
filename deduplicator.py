#!/usr/bin/env python3
"""OmniForge dataset deduplicator using SHA256 exact hash deduplication.

Source: ZIP1 (primary). Uses SHA256 for exact content deduplication.
"""

import gzip
import json
import hashlib
import sys

import config


def deduplicate() -> None:
    """Remove exact duplicate documents using SHA256 hashing."""
    if not config.CLEAN_DATASET_PATH.exists():
        print(f"[dedup] ERROR: Clean dataset not found at {config.CLEAN_DATASET_PATH}")
        print("[dedup] Run dataset_cleaner.py first.")
        sys.exit(1)

    config.CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen_hashes = set()
    total = 0
    duplicates = 0

    with gzip.open(config.CLEAN_DATASET_PATH, "rt", encoding="utf-8") as fin, \
         gzip.open(config.DEDUPED_DATASET_PATH, "wt", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            content = record.get("content", "")
            doc_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            if doc_hash in seen_hashes:
                duplicates += 1
                continue

            seen_hashes.add(doc_hash)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    written = total - duplicates
    print(f"[dedup] Deduplication complete.")
    print(f"[dedup]   Read:        {total:>8,}")
    print(f"[dedup]   Duplicates:  {duplicates:>8,}")
    print(f"[dedup]   Written:     {written:>8,}")
    print(f"[dedup]   Reduction:   {100 * duplicates / max(total, 1):.1f}%")


if __name__ == "__main__":
    deduplicate()