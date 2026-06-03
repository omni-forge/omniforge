#!/usr/bin/env python3
"""OmniForge dataset collector - downloads coding data from Hugging Face.

Streams bigcode/the-stack-dedup, filters to Python and JavaScript,
writes each document as a JSON line to raw_dataset.jsonl.gz.
Includes retry logic with exponential backoff on connection errors.
"""

import argparse
import gzip
import json
import random
import sys
import time
from typing import Dict, Iterator, Optional

import requests
from datasets import load_dataset

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Python and JavaScript code data for OmniForge.")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum total documents to collect (for testing).")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path (default: from config).")
    return parser.parse_args()


def extract_text(example: Dict) -> str:
    """Extract code text from a dataset example, trying multiple field names."""
    for key in ("content", "text", "code"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def stream_language(language: str):
    """Stream examples for one language subset."""
    return load_dataset(
        config.DATASET_SOURCE,
        data_dir=language,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )


def retrying_language_iterator(language: str) -> Iterator[Dict]:
    """Iterate over a language subset with automatic retry and exponential backoff."""
    retries = 0
    yielded = 0
    while retries <= config.COLLECTOR_MAX_RETRIES:
        try:
            dataset = stream_language(language)
            for index, example in enumerate(dataset):
                # Skip already-yielded examples after a reconnect
                if index < yielded:
                    continue
                yielded += 1
                yield example
            return  # exhausted cleanly
        except (requests.RequestException, ConnectionError, TimeoutError, OSError) as exc:
            retries += 1
            if retries > config.COLLECTOR_MAX_RETRIES:
                print(
                    f"[collector] ERROR: exceeded max retries ({config.COLLECTOR_MAX_RETRIES}) "
                    f"for language '{language}': {exc}",
                    file=sys.stderr,
                )
                raise
            sleep_seconds = config.COLLECTOR_INITIAL_BACKOFF_SECONDS * (2 ** (retries - 1))
            sleep_seconds += random.uniform(0.0, 1.0)  # jitter
            print(
                f"[collector] WARNING: connection error for '{language}': {exc}. "
                f"Retry {retries}/{config.COLLECTOR_MAX_RETRIES} in {sleep_seconds:.1f}s.",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()
    config.ensure_directories()

    output_path = config.RAW_DATASET_PATH if args.output is None else config.PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_docs: Optional[int] = args.max_docs
    total_docs = 0
    total_bytes = 0

    print(f"[collector] Dataset source : {config.DATASET_SOURCE}")
    print(f"[collector] Languages      : {', '.join(config.DATASET_LANGUAGES)}")
    print(f"[collector] Max documents  : {'unlimited' if max_docs is None else f'{max_docs:,}'}")
    print(f"[collector] Output         : {output_path}")

    with gzip.open(output_path, "wt", encoding="utf-8") as writer:
        # Round-robin across languages so the dataset stays balanced
        language_iters = {lang: retrying_language_iterator(lang) for lang in config.DATASET_LANGUAGES}
        active_languages = list(language_iters.keys())
        lang_index = 0

        while active_languages:
            if max_docs is not None and total_docs >= max_docs:
                break

            language = active_languages[lang_index % len(active_languages)]
            try:
                example = next(language_iters[language])
            except StopIteration:
                active_languages.remove(language)
                if not active_languages:
                    break
                lang_index %= len(active_languages)
                continue

            text = extract_text(example)
            if not text:
                lang_index += 1
                continue

            record = {
                "content": text,
                "language": language.lower(),
                "source": config.DATASET_SOURCE,
                "path": example.get("path") or example.get("max_stars_repo_path") or "",
                "repository": example.get("repository_name") or example.get("repo_name") or "",
                "license": example.get("license") or "",
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            writer.write(line)
            total_docs += 1
            total_bytes += len(line.encode("utf-8"))

            if total_docs % config.COLLECTOR_LOG_INTERVAL == 0:
                print(
                    f"[collector] Collected {total_docs:,} documents; "
                    f"{total_bytes / (1024 ** 2):.2f} MB uncompressed written.",
                    flush=True,
                )

            lang_index += 1

    print("[collector] ── Final Summary ──────────────────────────")
    print(f"[collector] Total documents collected : {total_docs:,}")
    print(f"[collector] Total uncompressed bytes  : {total_bytes:,} ({total_bytes / (1024**2):.2f} MB)")
    print(f"[collector] Output file               : {output_path}")


if __name__ == "__main__":
    main()
