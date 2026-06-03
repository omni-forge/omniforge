#!/usr/bin/env python3
import os, sys, subprocess, shutil
from pathlib import Path

GDRIVE_REMOTE = "gdrive:omniforge"
LOCAL_OMNIFORGE = "/kaggle/working/omniforge"
REPO_URL = "https://github.com/omni-forge/omniforge.git"

# Load rclone config from Kaggle Secret using official API
def load_rclone_conf():
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        conf = user_secrets.get_secret("RCLONE_CONF")
        if conf:
            print("[rclone] Loading config from Kaggle Secret API")
            return conf
    except Exception as e:
        print(f"[rclone] Kaggle Secret API error: {e}")
    
    # Fallback: try file path
    kaggle_secret = Path("/kaggle/secrets/RCLONE_CONF")
    if kaggle_secret.exists():
        print("[rclone] Loading config from Kaggle Secret file")
        return kaggle_secret.read_text()
    
    raise RuntimeError(
        "Kaggle Secret 'RCLONE_CONF' not found.\n"
        "Please add it in Kaggle: Add-ons -> Secrets -> 'RCLONE_CONF'\n"
        "Paste your rclone.conf content there."
    )

RCLONE_CONF = load_rclone_conf()

def run(cmd, check=True):
    print(f"\n[run] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[run] ERROR (exit {result.returncode}): {result.stderr.strip()}")
        if check:
            print("[run] Continuing despite error...")
    else:
        if result.stdout.strip():
            print(f"[run] {result.stdout.strip()}")
    return result.returncode == 0

def rclone(cmd, check=True):
    print(f"\n[rclone] {cmd}")
    result = subprocess.run(f"rclone {cmd}", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[rclone] ERROR (exit {result.returncode}): {result.stderr.strip()}")
        if check:
            print("[rclone] Continuing despite error...")
        return False
    if result.stdout.strip():
        print(f"[rclone] {result.stdout.strip()}")
    return True

def verify_rclone():
    """Test rclone connectivity before proceeding."""
    print("\n" + "="*60)
    print("VERIFYING RCLONE CONNECTIVITY")
    print("="*60)
    result = subprocess.run("rclone lsd gdrive: --max-depth 1", 
                          shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[verify] FAILED: {result.stderr.strip()}")
        print("[verify] Check your rclone config — token may be expired!")
        return False
    print("[verify] SUCCESS: Connected to Google Drive")
    print(f"[verify] Output: {result.stdout.strip()}")
    return True

def install_dependencies():
    print("\n" + "="*60)
    print("STEP 1: Installing dependencies")
    print("="*60)
    run("pip install -q numpy torch==2.2.0 transformers tokenizers datasets tqdm fastapi uvicorn sentencepiece huggingface-hub accelerate safetensors requests", check=False)
    run("curl https://rclone.org/install.sh | sudo bash || apt-get install -y rclone", check=False)

def setup_rclone():
    print("\n" + "="*60)
    print("STEP 2: Setting up rclone")
    print("="*60)
    
    conf_path = Path("/root/.config/rclone/rclone.conf")
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(RCLONE_CONF)
    print("[rclone] Config written.")
    
    if not verify_rclone():
        print("[rclone] WARNING: Drive not accessible — checkpoints will be LOCAL ONLY!")
    
    # Create remote directories (ignore errors if they exist)
    rclone("mkdir gdrive:omniforge/checkpoints", check=False)
    rclone("mkdir gdrive:omniforge/tokenizer", check=False)
    rclone("mkdir gdrive:omniforge/data/tokenized", check=False)
    rclone("mkdir gdrive:omniforge/logs", check=False)

def setup_project():
    print("\n" + "="*60)
    print("STEP 3: Setting up project")
    print("="*60)
    if Path(LOCAL_OMNIFORGE).exists():
        shutil.rmtree(LOCAL_OMNIFORGE)
    run(f"git clone {REPO_URL} {LOCAL_OMNIFORGE}", check=True)
    os.chdir(LOCAL_OMNIFORGE)

def restore_from_drive():
    print("\n" + "="*60)
    print("STEP 4: Restoring saved progress from Google Drive")
    print("="*60)
    os.makedirs(f"{LOCAL_OMNIFORGE}/checkpoints", exist_ok=True)
    os.makedirs(f"{LOCAL_OMNIFORGE}/tokenizer", exist_ok=True)
    os.makedirs(f"{LOCAL_OMNIFORGE}/data/tokenized", exist_ok=True)
    os.makedirs(f"{LOCAL_OMNIFORGE}/logs", exist_ok=True)
    
    # Download checkpoints
    rclone(f"copy gdrive:omniforge/checkpoints {LOCAL_OMNIFORGE}/checkpoints --transfers=4", check=False)
    ckpts = list(Path(f"{LOCAL_OMNIFORGE}/checkpoints").glob("checkpoint_step_*.pt"))
    if ckpts:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
        print(f"[restore] Found {len(ckpts)} checkpoint(s). Latest: {latest.name}")
    else:
        print("[restore] No checkpoints found. Starting fresh.")
    
    # Download tokenizer
    rclone(f"copy gdrive:omniforge/tokenizer {LOCAL_OMNIFORGE}/tokenizer --transfers=4", check=False)
    
    # Download tokenized data (only if missing)
    for fname in ["train.bin", "val.bin", "test.bin"]:
        dst = f"{LOCAL_OMNIFORGE}/data/tokenized/{fname}"
        if not Path(dst).exists() or Path(dst).stat().st_size < 1000:
            rclone(f"copy gdrive:omniforge/data/tokenized/{fname} {LOCAL_OMNIFORGE}/data/tokenized/", check=False)
    
    # Download logs
    rclone(f"copy gdrive:omniforge/logs/training_log.csv {LOCAL_OMNIFORGE}/logs/", check=False)

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
    rclone(f"copy {LOCAL_OMNIFORGE}/tokenizer gdrive:omniforge/tokenizer --transfers=4", check=False)
    run("python prepare_dataset.py")
    for fname in ["train.bin", "val.bin", "test.bin"]:
        src = f"{LOCAL_OMNIFORGE}/data/tokenized/{fname}"
        if Path(src).exists():
            rclone(f"copy {src} gdrive:omniforge/data/tokenized/", check=False)

def run_training():
    print("\n" + "="*60)
    print("STEP 6: Training")
    print("="*60)
    result = subprocess.run("python train.py", shell=True)
    print(f"[train] Training exited with code: {result.returncode}")
    return result.returncode

def save_final_state():
    print("\n" + "="*60)
    print("STEP 7: Saving final state to Google Drive")
    print("="*60)
    
    # Upload any remaining checkpoints (should already be uploaded by fixed train.py)
    rclone(f"copy {LOCAL_OMNIFORGE}/checkpoints gdrive:omniforge/checkpoints --transfers=4", check=False)
    rclone(f"copy {LOCAL_OMNIFORGE}/logs/training_log.csv gdrive:omniforge/logs/", check=False)
    
    # Verify what we have on Drive
    result = subprocess.run("rclone lsf gdrive:omniforge/checkpoints/ | sort", 
                          shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        files = result.stdout.strip().split("\n")
        print(f"[save] {len(files)} file(s) on Drive checkpoints/")
        for f in files[-5:]:  # Show last 5
            print(f"[save]   {f}")
    
    print("[save] Final save complete.")

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
    exit_code = run_training()
    save_final_state()
    
    print("\n" + "="*60)
    if exit_code == 0:
        print("Training completed successfully!")
    else:
        print(f"Training exited with code {exit_code} — progress saved to Drive.")
    print("Next session triggers automatically tomorrow.")
    print("="*60)

if __name__ == "__main__":
    main()
