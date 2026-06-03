#!/usr/bin/env python3
"""OmniForge inference utilities and streaming CLI.

Source: ZIP1 (primary). Robust streaming with top-k/top-p/greedy.
"""

import argparse
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import torch
from transformers import AutoTokenizer

import config
from model import ModelConfig, OmniForge


_MODEL: Optional[OmniForge] = None
_TOKENIZER = None
_DEVICE: Optional[torch.device] = None


def latest_checkpoint() -> Path:
    checkpoints = sorted(config.CHECKPOINT_DIR.glob("checkpoint_step_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {config.CHECKPOINT_DIR}. Train a model first.")
    return checkpoints[-1]


def load_model_and_tokenizer() -> Tuple[OmniForge, object, torch.device]:
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None and _TOKENIZER is not None and _DEVICE is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(config.TOKENIZER_OUTPUT_DIR), use_fast=True)
    model = OmniForge.from_config().to(device)
    checkpoint_path = latest_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[inference] Loaded checkpoint: {checkpoint_path}")
    print(f"[inference] Device: {device}")
    _MODEL, _TOKENIZER, _DEVICE = model, tokenizer, device
    return model, tokenizer, device


def encode_prompt(prompt: str, tokenizer, device: torch.device) -> torch.Tensor:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > config.CONTEXT_LENGTH:
        ids = ids[-config.CONTEXT_LENGTH:]
    if not ids:
        ids = [config.BOS_TOKEN_ID]
    return torch.tensor([ids], dtype=torch.long, device=device)


def sample_next_token(logits: torch.Tensor, strategy: str, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    strategy = strategy.lower().strip()
    if strategy == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)
    temperature = max(float(temperature), 1e-6)
    logits = logits / temperature
    if strategy in {"top_k", "top-k", "topk"} or top_k > 0:
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], torch.finfo(logits.dtype).min)
    if strategy in {"top_p", "top-p", "topp", "nucleus"} or top_p < 1.0:
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            mask = cumulative > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, torch.finfo(sorted_logits.dtype).min)
            logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
            logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_tokens(
    prompt: str,
    max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
    strategy: str = "top_p",
    temperature: float = config.DEFAULT_TEMPERATURE,
    top_k: int = config.DEFAULT_TOP_K,
    top_p: float = config.DEFAULT_TOP_P,
) -> Generator[str, None, None]:
    model, tokenizer, device = load_model_and_tokenizer()
    input_ids = encode_prompt(prompt, tokenizer, device)
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -config.CONTEXT_LENGTH:]
        logits = model(idx_cond)[:, -1, :]
        next_token = sample_next_token(logits, strategy, temperature, top_k, top_p)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        token_id = int(next_token.item())
        if token_id in {config.EOS_TOKEN_ID, config.EOD_TOKEN_ID}:
            break
        yield tokenizer.decode([token_id], skip_special_tokens=False)


def generate(
    prompt: str,
    max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
    strategy: str = "top_p",
    temperature: float = config.DEFAULT_TEMPERATURE,
    top_k: int = config.DEFAULT_TOP_K,
    top_p: float = config.DEFAULT_TOP_P,
) -> str:
    return "".join(generate_tokens(prompt, max_new_tokens, strategy, temperature, top_k, top_p))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniForge inference.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=config.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--strategy", type=str, default="top_p", choices=["greedy", "top_k", "top_p", "sample"])
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=config.DEFAULT_TOP_P)
    args = parser.parse_args()

    start = time.time()
    count = 0
    for token in generate_tokens(args.prompt, args.max_new_tokens, args.strategy, args.temperature, args.top_k, args.top_p):
        print(token, end="", flush=True)
        count += 1
    elapsed = max(time.time() - start, 1e-9)
    print(f"\n[inference] Generated {count} tokens in {elapsed:.2f}s ({count / elapsed:.2f} tokens/s)")


if __name__ == "__main__":
    main()