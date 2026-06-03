#!/usr/bin/env python3
"""Evaluate OmniForge perplexity and code completion executability.

Source: ZIP1 (primary). Calculates perplexity on test set and runs
        20 code completion tests using exec() in sandboxed namespace.
"""

import math
import traceback
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import config
from inference import generate, load_model_and_tokenizer
from train import get_batch, load_memmap


STUBS = [
    "def add(a, b):\n    ",
    "def factorial(n):\n    ",
    "def fibonacci(n):\n    ",
    "def reverse_string(s):\n    ",
    "def is_prime(n):\n    ",
    "def flatten(items):\n    ",
    "def count_words(text):\n    ",
    "def merge_dicts(a, b):\n    ",
    "def safe_divide(a, b):\n    ",
    "def unique_values(values):\n    ",
    "def read_lines(path):\n    ",
    "def write_text(path, text):\n    ",
    "def normalize_whitespace(text):\n    ",
    "def chunk_list(items, size):\n    ",
    "def binary_search(items, target):\n    ",
    "def transpose(matrix):\n    ",
    "def sort_by_key(rows, key):\n    ",
    "def parse_int(value, default=0):\n    ",
    "def group_by_first_letter(words):\n    ",
    "def clamp(value, low, high):\n    ",
]


def calculate_perplexity(model, test_data: np.memmap, device: torch.device,
                         batches: int = 50) -> float:
    """Calculate perplexity on the test set."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(test_data, config.BATCH_SIZE, device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=config.PAD_TOKEN_ID,
            )
            losses.append(loss.item())
    model.train()
    avg_loss = float(np.mean(losses))
    # Cap before exp() to avoid OverflowError on high-loss untrained models
    perplexity = float(math.exp(min(avg_loss, 20.0)))
    return perplexity


def test_code_completion(model, tokenizer, device: torch.device,
                         max_new_tokens: int = 50) -> List[Dict]:
    """Test code completions and check if they execute without error."""
    results = []
    for stub in STUBS:
        full_completion = generate(
            stub,
            max_new_tokens=max_new_tokens,
            strategy="greedy",
            temperature=0.1,
            top_k=50,
            top_p=0.95,
        )
        # Reconstruct the full function
        full_code = stub + full_completion
        # Attempt to parse/execute
        sandbox = {}
        passed = False
        error = None
        try:
            exec(full_code, sandbox)
            passed = True
        except Exception as e:
            error = traceback.format_exc()

        results.append({
            "stub": stub,
            "completion": full_completion,
            "passed": passed,
            "error": error,
        })

    return results


def main() -> None:
    print("=" * 60)
    print("OmniForge Evaluation")
    print("=" * 60)

    # Load model
    model, tokenizer, device = load_model_and_tokenizer()

    # 1. Perplexity
    print("\n[1/2] Calculating perplexity on test set...")
    test_data = load_memmap(config.TEST_BIN_PATH)
    ppl = calculate_perplexity(model, test_data, device)
    print(f"  Perplexity: {ppl:.2f}")

    # 2. Code completion test
    print("\n[2/2] Testing code completions (20 stubs)...")
    results = test_code_completion(model, tokenizer, device)

    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n  Passing: {passed_count}/20")

    for i, r in enumerate(results):
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] Test {i + 1}: {r['stub'][:50].strip()}")

    print("\n" + "=" * 60)
    print(f"Final Score: {passed_count}/20")
    print(f"Perplexity: {ppl:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()