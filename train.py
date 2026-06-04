#!/usr/bin/env python3
"""OmniForge Fine-tuning Script - Fine-tunes TinyLlama-1.1B on code data."""

import csv, math, random, time, subprocess
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT   = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"
DATA_DIR       = PROJECT_ROOT / "data" / "tokenized"
TRAIN_BIN      = DATA_DIR / "train.bin"
VAL_BIN        = DATA_DIR / "val.bin"
LOG_PATH       = LOG_DIR / "training_log.csv"

MODEL_NAME     = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
CONTEXT_LENGTH = 2048
BATCH_SIZE     = 2
GRAD_ACCUM     = 32
LEARNING_RATE  = 2e-5
MIN_LR         = 1e-6
WARMUP_STEPS   = 200
MAX_STEPS      = 10000
EVAL_INTERVAL  = 200
SAVE_INTERVAL  = 200
GRAD_CLIP      = 1.0
SEED           = 1337
TOKEN_DTYPE    = np.uint16
GDRIVE_CKPT    = "gdrive:omniforge/checkpoints"

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_memmap(path):
    if not path.exists(): raise FileNotFoundError(f"Missing: {path}")
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")

def get_batch(data, batch_size, device):
    max_start = len(data) - CONTEXT_LENGTH - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = np.stack([np.asarray(data[i:i+CONTEXT_LENGTH], dtype=np.int64) for i in starts])
    y = np.stack([np.asarray(data[i+1:i+1+CONTEXT_LENGTH], dtype=np.int64) for i in starts])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)

def lr_schedule(step):
    if step < WARMUP_STEPS: return LEARNING_RATE * (step+1) / max(1, WARMUP_STEPS)
    if step >= MAX_STEPS: return MIN_LR
    ratio = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5*(1.0+math.cos(math.pi*ratio))*(LEARNING_RATE-MIN_LR)

def latest_checkpoint():
    ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts[-1] if ckpts else None

def save_checkpoint(model, optimizer, scaler, step, loss):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"checkpoint_step_{step}.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(), "step": step, "loss": float(loss)}, path)
    print(f"[train] Saved checkpoint: {path.name}")
    result = subprocess.run(f"rclone copy {path} {GDRIVE_CKPT}/", shell=True, capture_output=True, text=True)
    if result.returncode != 0: print(f"[train] UPLOAD ERROR: {result.stderr.strip()}")
    else: print(f"[train] Uploaded to Drive: {path.name}")
    all_ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for old in all_ckpts[:-3]:
        subprocess.run(f"rclone delete {GDRIVE_CKPT}/{old.name}", shell=True, capture_output=True)
        old.unlink()
    return path

def load_checkpoint(model, optimizer, scaler, device):
    ckpt_path = latest_checkpoint()
    if ckpt_path is None: return 0, float("nan")
    print(f"[train] Resuming from: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt: scaler.load_state_dict(ckpt["scaler_state_dict"])
    return int(ckpt.get("step", 0)), float(ckpt.get("loss", float("nan")))

@torch.no_grad()
def estimate_loss(model, val_data, device, eval_iters=10):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_data, BATCH_SIZE, device)
        with autocast(device_type=device.type, dtype=torch.float16):
            out = model(input_ids=x, labels=y)
            losses.append(out.loss.item())
    model.train()
    return float(np.mean(losses))

def init_log():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["step","train_loss","val_loss","lr","sps","eta_h"])

def append_log(step, train_loss, val_loss, lr, sps, eta_h):
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([step, train_loss, val_loss or "", lr, sps, eta_h])

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] Loading {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    print(f"[train] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_data = load_memmap(TRAIN_BIN)
    val_data = load_memmap(VAL_BIN)
    print(f"[train] Train tokens: {len(train_data):,}")
    print(f"[train] Val tokens: {len(val_data):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.95), weight_decay=0.1)
    scaler = GradScaler(device_type="cuda", enabled=True)
    start_step, _ = load_checkpoint(model, optimizer, scaler, device)
    init_log()
    model.train()
    t0 = time.time()
    for step in range(start_step+1, MAX_STEPS+1):
        lr = lr_schedule(step)
        for pg in optimizer.param_groups: pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = get_batch(train_data, BATCH_SIZE, device)
            with autocast(device_type=device.type, dtype=torch.float16):
                out = model(input_ids=x, labels=y)
                loss = out.loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            accum_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        elapsed = max(time.time()-t0, 1e-9)
        sps = (step-start_step)/elapsed
        eta_h = (MAX_STEPS-step)/max(sps,1e-9)/3600
        val_loss = None
        if step % EVAL_INTERVAL == 0: val_loss = estimate_loss(model, val_data, device)
        val_txt = f" val_loss={val_loss:.4f}" if val_loss else ""
        print(f"[train] step={step:,}/{MAX_STEPS:,} loss={accum_loss:.4f}{val_txt} lr={lr:.2e} sps={sps:.3f} eta={eta_h:.2f}h")
        append_log(step, accum_loss, val_loss, lr, sps, eta_h)
        if step % SAVE_INTERVAL == 0: save_checkpoint(model, optimizer, scaler, step, accum_loss)
    save_checkpoint(model, optimizer, scaler, MAX_STEPS, accum_loss)
    print("[train] Fine-tuning complete.")

if __name__ == "__main__":
    main()
