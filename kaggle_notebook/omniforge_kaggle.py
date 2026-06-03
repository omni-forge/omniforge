#!/usr/bin/env python3
import os, sys, subprocess, shutil
from pathlib import Path

GDRIVE_REMOTE = "gdrive:omniforge"
LOCAL_OMNIFORGE = "/kaggle/working/omniforge"
REPO_URL = "https://github.com/omni-forge/omniforge.git"

# Load rclone config from Kaggle Secret (recommended) or fallback to embedded
def load_rclone_conf():
    kaggle_secret = Path("/kaggle/secrets/RCLONE_CONF")
    if kaggle_secret.exists():
        print("[rclone] Loading config from Kaggle Secret")
        return kaggle_secret.read_text()
    print("[rclone] WARNING: No Kaggle Secret found. Using embedded config (may be expired)")
    return """[gdrive]
type = drive
scope = drive
team_drive =
"""

RCLONE_CONF = load_rclone_conf()

def run(cmd):
    print(f"\n[run] {cmd}")
    subprocess.run(cmd, shell=True)

def rclone(cmd):
    print(f"\n[rclone] {cmd}")
    subprocess.run(f"rclone {cmd}", shell=True)

def install_dependencies():
    print("\n" + "="*60)
    print("STEP 1: Installing dependencies")
    print("="*60)
    run("pip install -q numpy==1.26.4 torch==2.2.0 transformers tokenizers datasets tqdm fastapi uvicorn sentencepiece huggingface-hub accelerate safetensors requests")
    run("curl https://rclone.org/install.sh | sudo bash || apt-get install -y rclone")

def setup_rclone():
    print("\n" + "="*60)
    print("STEP 2: Setting up rclone")
    print("="*60)
    conf_path = Path("/root/.config/rclone/rclone.conf")
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(RCLONE_CONF)
    print("[rclone] Config written.")
    result = subprocess.run("rclone lsd gdrive: --max-depth 1", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("[rclone] WARNING: Could not connect to Google Drive.")
        print(result.stderr.strip())
    else:
        print("[rclone] Google Drive connected successfully!")
    rclone("mkdir gdrive:omniforge/checkpoints")
    rclone("mkdir gdrive:omniforge/tokenizer")
    rclone("mkdir gdrive:omniforge/data/tokenized")
    rclone("mkdir gdrive:omniforge/logs")

def setup_project():
    print("\n" + "="*60)
    print("STEP 3: Setting up project")
    print("="*60)
    if Path(LOCAL_OMNIFORGE).exists():
        shutil.rmtree(LOCAL_OMNIFORGE)
    run(f"git clone {REPO_URL} {LOCAL_OMNIFORGE}")
    os.chdir(LOCAL_OMNIFORGE)

def restore_from_drive():
    print("\n" + "="*60)
    print("STEP 4: Restoring saved progress from Google Drive")
    print("="*60)
    os.makedirs(f"{LOCAL_OMNIFORGE}/checkpoints", exist_ok=True)
    os.makedirs(f"{LOCAL_OMNIFORGE}/tokenizer", exist_ok=True)
    os.makedirs(f"{LOCAL_OMNIFORGE}/data/tokenized", exist_ok=True)
    os.makedirs(f"{LOCAL_OMNIFORGE}/logs", exist_ok=True)
    rclone(f"copy gdrive:omniforge/checkpoints {LOCAL_OMNIFORGE}/checkpoints --transfers=4")
    ckpts = list(Path(f"{LOCAL_OMNIFORGE}/checkpoints").glob("checkpoint_step_*.pt"))
    if ckpts:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
        print(f"[restore] Resuming from: {latest.name}")
    else:
        print("[restore] No checkpoints found. Starting fresh.")
    rclone(f"copy gdrive:omniforge/tokenizer {LOCAL_OMNIFORGE}/tokenizer --transfers=4")
    for fname in ["train.bin", "val.bin", "test.bin"]:
        dst = f"{LOCAL_OMNIFORGE}/data/tokenized/{fname}"
        if not Path(dst).exists():
            rclone(f"copy gdrive:omniforge/data/tokenized/{fname} {LOCAL_OMNIFORGE}/data/tokenized/")
    rclone(f"copy gdrive:omniforge/logs/training_log.csv {LOCAL_OMNIFORGE}/logs/")

def run_data_pipeline():
    train_bin = Path(f"{LOCAL_OMNIFORGE}/data/tokenized/train.bin")
    if train_bin.exists() and train_bin.stat().st_size > 100:
        print("\n[pipeline] train.bin exists. Skipping data pipeline.")
        return
    print("\n" + "="*60)
    print("STEP 5: Running data pipeline (first time only)")
    print("="*60)
    run("python dataset_collector.py --max-docs 500000")
    run("python dataset_cleaner.py")
    run("python deduplicator.py")
    run("python train_tokenizer.py")
    rclone(f"copy {LOCAL_OMNIFORGE}/tokenizer gdrive:omniforge/tokenizer --transfers=4")
    run("python prepare_dataset.py")
    for fname in ["train.bin", "val.bin", "test.bin"]:
        src = f"{LOCAL_OMNIFORGE}/data/tokenized/{fname}"
        if Path(src).exists():
            rclone(f"copy {src} gdrive:omniforge/data/tokenized/")

def run_training():
    print("\n" + "="*60)
    print("STEP 6: Training")
    print("="*60)
    subprocess.run("python train.py", shell=True)

def save_final_state():
    print("\n" + "="*60)
    print("STEP 7: Saving final state to Google Drive")
    print("="*60)
    rclone(f"copy {LOCAL_OMNIFORGE}/checkpoints gdrive:omniforge/checkpoints --transfers=4")
    rclone(f"copy {LOCAL_OMNIFORGE}/logs/training_log.csv gdrive:omniforge/logs/")
    print("[save] All progress saved to Google Drive.")

def main():
    print("="*60)
    print("  OmniForge Automated Training on Kaggle")
    print("  Triggered by GitHub Actions")
    print("="*60)
    install_dependencies()
    setup_rclone()
    setup_project()
    restore_from_drive()
    run_data_pipeline()
    run_training()
    save_final_state()
    print("\n" + "="*60)
    print("Session complete. Progress saved to Google Drive.")
    print("Next session triggers automatically tomorrow.")
    print("="*60)

if __name__ == "__main__":
    main()
