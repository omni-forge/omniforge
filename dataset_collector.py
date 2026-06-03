#!/usr/bin/env python3
"""Dataset collector using free, publicly available code datasets."""
import gzip, json, os, argparse
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "raw_dataset.jsonl.gz"
MAX_DOCS = 500_000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=MAX_DOCS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[collector] Using free CodeSearchNet dataset")
    print(f"[collector] Max documents : {args.max_docs:,}")
    print(f"[collector] Output         : {OUTPUT_FILE}")

    count = 0
    with gzip.open(OUTPUT_FILE, "wt", encoding="utf-8") as f:
        for lang in ["python", "javascript", "java", "go", "ruby", "php"]:
            if count >= args.max_docs:
                break
            try:
                print(f"[collector] Loading {lang}...")
                ds = load_dataset("code_search_net", lang, split="train", trust_remote_code=False)
                for example in ds:
                    if count >= args.max_docs:
                        break
                    code = example.get("whole_func_string", "")
                    if code and len(code) > 50:
                        f.write(json.dumps({"code": code, "language": lang}) + "\n")
                        count += 1
                        if count % 10000 == 0:
                            print(f"[collector] Collected {count:,} documents...")
            except Exception as e:
                print(f"[collector] WARNING: skipping {lang}: {e}")
                continue

    print(f"[collector] Done. Total documents: {count:,}")

if __name__ == "__main__":
    main()
