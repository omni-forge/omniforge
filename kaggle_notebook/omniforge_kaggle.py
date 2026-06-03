#!/usr/bin/env python3
import os, sys, subprocess, shutil
from pathlib import Path

GDRIVE_OMNIFORGE = "/root/gdrive/MyDrive/omniforge"
LOCAL_OMNIFORGE  = "/kaggle/working/omniforge"
REPO_URL         = "https://github.com/omni-forge/omniforge.git"

RCLONE_CONF = """[gdrive]
type = drive
scope = drive
token = {"access_token":"ya29.a0AQvPyIM-zcLlltNq6xkvleBuYx3Laae59iGD0Sx8kkIVBKE7Os7mk-kJDuUJgJTrTPlPB6SYqgDH5G5j1lGwnvQmxIak3JyzC4rKLTz8die6AQBqSFHXnetDW17UNBs1ZkiJs5hHCF0RsvL2hG0uYGmZRUf4fqgY6SD4b7pzgH8va89r4_sDaJmxUY289iQ2hvAk_b0aCgYKAQESARESFQHGX2MippQVMQSGSjTiHVPMnki0FA0206","token_type":"Bearer","refresh_token":"1//03gw5zkUG7lI3CgYIARAAGAMSNwF-L9IrRlauRoGMRfiOXhc06fGu0EBtmnx-wsrTzQ4i191jgEeSd9vJGk3KpHIOMEO732hurDM","expiry":"2026-06-03T04:14:10.029692514Z","expires_in":3599}
team_drive =
"""

def run(cmd):
    print(f"\n[run] {cmd}")
    subprocess.run(cmd, shell=True)

def install_dependencies():
    print("\n" + "="*60)
    print("STEP 1: Installing dependencies")
    print("="*60)
    run("pip install -q torch transformers tokenizers datasets tqdm fastapi uvicorn sentencepiece huggingface-hub accelerate safetensors requests")
    run("curl https://rclone.org/install.sh | sudo bash || apt-get install -y rclone")

def mount_google_drive():
    print("\n" + "="*60)
    print("STEP 2: Mounting Google Drive via rclone")
    print("="*60)
    os.makedirs("/root/gdrive", exist_ok=True)
    conf_path = Path("/root/.config/rclone/rclone.conf")
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(RCLONE_CONF)
    print("[drive] rclone.conf written from embedded config.")
    run("rclone mount gdrive: /root/gdrive --daemon --no-checksum --transfers=4 --buffer-size=256M &")
    import time
    time.sleep(10)
    if not Path("/root/gdrive/MyDrive").exists():
        print("[drive] ERROR: Google Drive did not mount correctly.")
        sys.exit(1)
    print("[drive] Google Drive mounted successfully.")
    os.makedirs(GDRIVE_OMNIFORGE, exist_ok=True)
    os.makedirs(f"{GDRIVE_OMNIFORGE}/checkpoints", exist_ok=True)
    os.makedirs(f"{GDRIVE_OMNIFORGE}/data/tokenized", exist_ok=True)
    os.makedirs(f"{GDRIVE_OMNIFORGE}/tokenizer", exist_ok=True)
    os.makedirs(f"{GDRIVE_OMNIFORGE}/logs", exist_ok=True)

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
    drive_ckpt = f"{GDRIVE_OMNIFORGE}/checkpoints"
    local_ckpt = f"{LOCAL_OMNIFORGE}/checkpoints"
    os.makedirs(local_ckpt, exist_ok=True)
    if Path(drive_ckpt).exists():
        run(f"cp -r {drive_ckpt}/. {local_ckpt}/")
        ckpts = list(Path(local_ckpt).glob("checkpoint_step_*.pt"))
        if ckpts:
            latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
            print(f"[restore] Resuming from: {latest.name}")
        else:
            print("[restore] No checkpoints found. Starting fresh.")
    drive_tok = f"{GDRIVE_OMNIFORGE}/tokenizer"
    local_tok = f"{LOCAL_OMNIFORGE}/tokenizer"
    os.makedirs(local_tok, exist_ok=True)
    if Path(drive_tok).exists() and any(Path(drive_tok).iterdir()):
        run(f"cp -r {drive_tok}/. {local_tok}/")
    drive_data = f"{GDRIVE_OMNIFORGE}/data/tokenized"
    local_data = f"{LOCAL_OMNIFORGE}/data/tokenized"
    os.makedirs(local_data, exist_ok=True)
    for fname in ["train.bin", "val.bin", "test.bin"]:
        src = f"{drive_data}/{fname}"
        dst = f"{local_data}/{fname}"
        if Path(src).exists() and not Path(dst).exists():
            run(f"cp {src} {dst}")
    drive_log = f"{GDRIVE_OMNIFORGE}/logs/training_log.csv"
    local_log = f"{LOCAL_OMNIFORGE}/logs/training_log.csv"
    os.makedirs(f"{LOCAL_OMNIFORGE}/logs", exist_ok=True)
    if Path(drive_log).exists():
        run(f"cp {drive_log} {local_log}")

def run_data_pipeline():
    train_bin = Path(f"{LOCAL_OMNIFORGE}/data/tokenized/train.bin")
    if train_bin.exists():
        print("\n[pipeline] train.bin exists. Skipping data pipeline.")
        return
    print("\n" + "="*60)
    print("STEP 5: Running data pipeline (first time only)")
    print("="*60)
    run("python dataset_collector.py --max-docs 500000")
    run("python dataset_cleaner.py")
    run("python deduplicator.py")
    run("python train_tokenizer.py")
    run(f"cp -r tokenizer/. {GDRIVE_OMNIFORGE}/tokenizer/")
    run("python prepare_dataset.py")
    for fname in ["train.bin", "val.bin", "test.bin"]:
        src = f"data/tokenized/{fname}"
        dst = f"{GDRIVE_OMNIFORGE}/data/tokenized/{fname}"
        if Path(src).exists():
            run(f"cp {src} {dst}")

def run_training():
    print("\n" + "="*60)
    print("STEP 6: Training")
    print("="*60)
    subprocess.run("python train.py", shell=True)

def save_final_state():
    print("\n" + "="*60)
    print("STEP 7: Saving final state to Google Drive")
    print("="*60)
    run(f"cp -r checkpoints/. {GDRIVE_OMNIFORGE}/checkpoints/")
    run(f"cp logs/training_log.csv {GDRIVE_OMNIFORGE}/logs/ 2>/dev/null || true")

def main():
    print("="*60)
    print("  OmniForge Automated Training on Kaggle")
    print("  Triggered by GitHub Actions")
    print("="*60)
    install_dependencies()
    mount_google_drive()
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
