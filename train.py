#!/usr/bin/env python3
"""Train OmniForge from packed binary token datasets.

Source: ZIP1 (primary training loop) with ZIP6 improvements (bfloat16 detection).
Features: cosine LR schedule, gradient accumulation, mixed precision, checkpointing.
"""

import csv
import math
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import config
from model import OmniForge


TOKEN_DTYPE = np.uint16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_memmap(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}. Run prepare_dataset.py first.")
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def get_batch(data: np.memmap, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= config.CONTEXT_LENGTH + 1:
        raise ValueError("Dataset is too small for one training batch.")
    max_start = len(data) - config.CONTEXT_LENGTH - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = np.stack([np.asarray(data[i: i + config.CONTEXT_LENGTH], dtype=np.int64) for i in starts])
    y = np.stack([np.asarray(data[i + 1: i + 1 + config.CONTEXT_LENGTH], dtype=np.int64) for i in starts])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)


def learning_rate(step: int) -> float:
    if step < config.WARMUP_STEPS:
        return config.LEARNING_RATE * (step + 1) / max(1, config.WARMUP_STEPS)
    if step >= config.MAX_STEPS:
        return config.MIN_LEARNING_RATE
    decay_ratio = (step - config.WARMUP_STEPS) / max(1, config.MAX_STEPS - config.WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.MIN_LEARNING_RATE + coeff * (config.LEARNING_RATE - config.MIN_LEARNING_RATE)


def latest_checkpoint() -> Optional[Path]:
    checkpoints = sorted(config.CHECKPOINT_DIR.glob("checkpoint_step_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(model: OmniForge, optimizer: torch.optim.Optimizer,
                    scaler: GradScaler, step: int, loss: float) -> Path:
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CHECKPOINT_DIR / f"checkpoint_step_{step}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "loss": float(loss),
        "model_config": model.cfg.__dict__,
    }, path)
    print(f"[train] Saved checkpoint: {path}")
    import subprocess
    subprocess.run(f"rclone copy {path} gdrive:omniforge/checkpoints/", shell=True)
    return path


def load_checkpoint(model: OmniForge, optimizer: torch.optim.Optimizer,
                    scaler: GradScaler, device: torch.device) -> Tuple[int, float]:
    checkpoint_path = latest_checkpoint()
    if checkpoint_path is None:
        return 0, float("nan")
    print(f"[train] Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint.get("step", 0)), float(checkpoint.get("loss", float("nan")))


@torch.no_grad()
def estimate_loss(model: OmniForge, val_data: np.memmap, device: torch.device, eval_iters: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_data, config.BATCH_SIZE, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=config.PAD_TOKEN_ID)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def init_log() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not config.TRAINING_LOG_PATH.exists():
        with open(config.TRAINING_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "learning_rate", "steps_per_second", "eta_seconds"])


def append_log(step: int, train_loss: float, val_loss: Optional[float], lr: float, sps: float, eta: float) -> None:
    with open(config.TRAINING_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([step, train_loss, "" if val_loss is None else val_loss, lr, sps, eta])


def main() -> None:
    config.ensure_directories()
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    # Use bfloat16 if supported (better stability than float16)
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16 if use_amp else torch.float32

    print(f"[train] Device: {device}")
    print(f"[train] Mixed precision dtype: {amp_dtype}")

    train_data = load_memmap(config.TRAIN_BIN_PATH)
    val_data = load_memmap(config.VAL_BIN_PATH)
    print(f"[train] Train tokens: {len(train_data):,}")
    print(f"[train] Val tokens: {len(val_data):,}")

    model = OmniForge.from_config().to(device)
    model.count_parameters()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY,
    )
    scaler = GradScaler(enabled=use_amp)
    start_step, resume_loss = load_checkpoint(model, optimizer, scaler, device)
    init_log()

    model.train()
    t0 = time.time()
    running_loss = resume_loss if not math.isnan(resume_loss) else 0.0

    for step in range(start_step + 1, config.MAX_STEPS + 1):
        lr = learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.GRADIENT_ACCUMULATION_STEPS):
            x, y = get_batch(train_data, config.BATCH_SIZE, device)
            with autocast(dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                       ignore_index=config.PAD_TOKEN_ID)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        train_loss = accumulated_loss
        running_loss = train_loss
        now = time.time()
        elapsed_total = max(now - t0, 1e-9)
        trained_steps = step - start_step
        steps_per_second = trained_steps / elapsed_total
        eta_seconds = (config.MAX_STEPS - step) / max(steps_per_second, 1e-9)

        val_loss = None
        if step % config.EVAL_INTERVAL == 0:
            val_loss = estimate_loss(model, val_data, device)

        append_log(step, train_loss, val_loss, lr, steps_per_second, eta_seconds)
        val_text = "" if val_loss is None else f" val_loss={val_loss:.4f}"
        print(
            f"[train] step={step:,}/{config.MAX_STEPS:,} loss={train_loss:.4f}{val_text} "
            f"lr={lr:.6e} sps={steps_per_second:.3f} eta={eta_seconds / 3600:.2f}h"
        )

        if step % config.SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, scaler, step, running_loss)

    save_checkpoint(model, optimizer, scaler, config.MAX_STEPS, running_loss)
    print("[train] Training complete.")


if __name__ == "__main__":
    main()