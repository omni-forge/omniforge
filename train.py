#!/usr/bin/env python3
"""OmniForge Fine-tuning Script - Fine-tunes TinyLlama-1.1B on code data."""

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0"
import csv, math, random, time, subprocess, threading, sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

PROJECT_ROOT   = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"
DATA_DIR       = PROJECT_ROOT / "data" / "tokenized"
TRAIN_BIN      = DATA_DIR / "train.bin"
VAL_BIN        = DATA_DIR / "val.bin"
LOG_PATH       = LOG_DIR / "training_log.csv"
HF_MODEL_DIR   = PROJECT_ROOT / "hf_model"
MODEL_CACHE    = Path("/kaggle/working/model_cache")  # Persistent model cache

MODEL_NAME     = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
CONTEXT_LENGTH = 2048
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
LEARNING_RATE  = 2e-6
MIN_LR         = 1e-6
WARMUP_STEPS   = 200
MAX_STEPS      = 10000
EVAL_INTERVAL  = 200
SAVE_INTERVAL  = 200
GRAD_CLIP      = 1.0
SEED           = 1337
TOKEN_DTYPE    = np.uint16
GDRIVE_CKPT    = "gdrive:omniforge/checkpoints"
GDRIVE_MODEL   = "gdrive:omniforge/model_cache"

def heartbeat(stop_event):
    """Print heartbeat every 30 seconds to prevent idle timeout"""
    while not stop_event.is_set():
        print(f"[heartbeat] Session alive at {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(30)

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

def save_checkpoint(model, optimizer, step, loss):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"checkpoint_step_{step}.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "step": step, "loss": float(loss)}, path)
    print(f"[train] Saved checkpoint: {path.name}")
    result = subprocess.run(f"rclone copy {path} {GDRIVE_CKPT}/", shell=True, capture_output=True, text=True)
    if result.returncode != 0: print(f"[train] UPLOAD ERROR: {result.stderr.strip()}")
    else: print(f"[train] Uploaded to Drive: {path.name}")
    all_ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for old in all_ckpts[:-3]:
        subprocess.run(f"rclone delete {GDRIVE_CKPT}/{old.name}", shell=True, capture_output=True)
        old.unlink()
    return path

def load_checkpoint(model, optimizer, device):
    ckpt_path = latest_checkpoint()
    if ckpt_path is None: return 0, float("nan")
    print(f"[train] Starting fresh (old checkpoint was FP16, incompatible): {ckpt_path.name}")
    return 0, float("nan")

@torch.no_grad()
def estimate_loss(model, val_data, device, eval_iters=10):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_data, BATCH_SIZE, device)
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

def download_model_with_retry():
    """Download model with heartbeat and retry logic"""
    print(f"[model] Checking for cached model at {MODEL_CACHE}...")
    
    # Check if model exists in local cache
    if MODEL_CACHE.exists() and (MODEL_CACHE / "config.json").exists():
        print(f"[model] Found cached model! Using local copy.")
        return str(MODEL_CACHE)
    
    # Check if model exists on Google Drive
    print(f"[model] Checking Google Drive for cached model...")
    result = subprocess.run(
        f"rclone lsf {GDRIVE_MODEL}/config.json 2>/dev/null",
        shell=True, capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"[model] Found model on Drive! Downloading...")
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(f"rclone copy {GDRIVE_MODEL}/ {MODEL_CACHE}/ --transfers=4", shell=True)
        if (MODEL_CACHE / "config.json").exists():
            print(f"[model] Model restored from Drive!")
            return str(MODEL_CACHE)
    
    # Download from HuggingFace with heartbeat
    print(f"[model] Downloading from HuggingFace: {MODEL_NAME}")
    print(f"[model] This may take 5-10 minutes. Heartbeat active...")
    
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat, args=(stop_event,))
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    
    try:
        # Use snapshot_download for resume support
        local_path = snapshot_download(
            MODEL_NAME,
            resume_download=True,
            local_dir=str(MODEL_CACHE),
            local_dir_use_symlinks=False,
            tqdm_class=None  # Disable default tqdm, we have heartbeat
        )
        print(f"[model] Download complete: {local_path}")
        
        # Save to Google Drive for future sessions
        print(f"[model] Uploading model to Google Drive for caching...")
        subprocess.run(f"rclone copy {MODEL_CACHE}/ {GDRIVE_MODEL}/ --transfers=4", shell=True)
        print(f"[model] Model cached to Drive!")
        
        return local_path
        
    except Exception as e:
        print(f"[model] ERROR downloading model: {e}")
        raise
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    
    # Download model with retry and caching
    model_path = download_model_with_retry()
    
    print(f"[train] Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    print(f"[train] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_data = load_memmap(TRAIN_BIN)
    val_data = load_memmap(VAL_BIN)
    print(f"[train] Train tokens: {len(train_data):,}")
    print(f"[train] Val tokens: {len(val_data):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler()
    start_step, _ = load_checkpoint(model, optimizer, device)
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
            out = model(input_ids=x, labels=y)
            loss = out.loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            accum_loss += loss.item()
        
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
        
        if step % SAVE_INTERVAL == 0: save_checkpoint(model, optimizer, step, accum_loss)
    
    save_checkpoint(model, optimizer, MAX_STEPS, accum_loss)
    
    print("[train] Saving model in HuggingFace format...")
    HF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(HF_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(HF_MODEL_DIR)
    print(f"[train] HuggingFace model saved to {HF_MODEL_DIR}")
    print("[train] Fine-tuning complete.")

if __name__ == "__main__":
    main()
