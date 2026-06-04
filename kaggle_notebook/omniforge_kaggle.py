#!/usr/bin/env python3
import os, sys, subprocess, shutil
from pathlib import Path

GDRIVE_REMOTE  = "gdrive:omniforge"
LOCAL_OMNIFORGE = "/kaggle/working/omniforge"
REPO_URL       = "https://github.com/omni-forge/omniforge.git"

def load_rclone_conf():
    try:
        from kaggle_secrets import UserSecretsClient
        conf = UserSecretsClient().get_secret("RCLONE_CONF")
        if conf:
            print("[rclone] Loaded from Kaggle Secret API")
            return conf
    except Exception as e:
        print(f"[rclone] Secret API error: {e}")
    fallback = Path("/kaggle/secrets/RCLONE_CONF")
    if fallback.exists():
        print("[rclone] Loaded from file")
        return fallback.read_text()
    raise RuntimeError("RCLONE_CONF secret not found. Add it in Kaggle Add-ons -> Secrets.")

RCLONE_CONF = load_rclone_conf()

def run(cmd, check=False):
    print(f"\n[run] {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip(): print(f"[run] {r.stdout.strip()}")
    if r.returncode != 0: print(f"[run] ERROR: {r.stderr.strip()}")
    return r.returncode == 0

def rclone(cmd, check=False):
    print(f"\n[rclone] {cmd}")
    r = subprocess.run(f"rclone {cmd}", shell=True, capture_output=True, text=True)
    if r.stdout.strip(): print(f"[rclone] {r.stdout.strip()}")
    if r.returncode != 0: print(f"[rclone] ERROR: {r.stderr.strip()}")
    return r.returncode == 0

def install_dependencies():
    print("\n" + "="*60)
    print("STEP 1: Installing dependencies")
    print("="*60)
    run("pip install -q numpy==1.26.4 torch==2.2.0 transformers tokenizers tqdm accelerate safetensors sentencepiece huggingface-hub")
    run("curl https://rclone.org/install.sh | sudo bash || apt-get install -y rclone", check=False)

def setup_rclone():
    print("\n" + "="*60)
    print("STEP 2: Setting up rclone")
    print("="*60)
    conf_path = Path("/root/.config/rclone/rclone.conf")
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(RCLONE_CONF.strip())
    print(f"[rclone] First 30 chars: {repr(RCLONE_CONF[:30])}")
    print("[rclone] Config written.")
    r = subprocess.run("rclone lsd gdrive: --max-depth 1", shell=True, capture_output=True, text=True)
    if r.returncode == 0:
        print("[rclone] Connected to Google Drive OK")
    else:
        print(f"[rclone] WARNING: Drive not accessible: {r.stderr.strip()}")
    rclone("mkdir gdrive:omniforge/checkpoints", check=False)
    rclone("mkdir gdrive:omniforge/data/tokenized", check=False)
    rclone("mkdir gdrive:omniforge/logs", check=False)

def setup_project():
    print("\n" + "="*60)
    print("STEP 3: Cloning repo")
    print("="*60)
    if Path(LOCAL_OMNIFORGE).exists():
        shutil.rmtree(LOCAL_OMNIFORGE)
    run(f"git clone {REPO_URL} {LOCAL_OMNIFORGE}")
    os.chdir(LOCAL_OMNIFORGE)
    print(f"[setup] Working dir: {os.getcwd()}")

def restore_from_drive():
    print("\n" + "="*60)
    print("STEP 4: Restoring from Google Drive")
    print("="*60)
    for d in ["checkpoints", "data/tokenized", "logs"]:
        os.makedirs(f"{LOCAL_OMNIFORGE}/{d}", exist_ok=True)
    rclone(f"copy gdrive:omniforge/checkpoints {LOCAL_OMNIFORGE}/checkpoints --transfers=4")
    ckpts = list(Path(f"{LOCAL_OMNIFORGE}/checkpoints").glob("checkpoint_step_*.pt"))
    if ckpts:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
        print(f"[restore] {len(ckpts)} checkpoint(s) found. Latest: {latest.name}")
    else:
        print("[restore] No checkpoints. Starting fresh.")
    for fname in ["train.bin", "val.bin", "test.bin"]:
        dst = Path(f"{LOCAL_OMNIFORGE}/data/tokenized/{fname}")
        if not dst.exists() or dst.stat().st_size < 1000:
            rclone(f"copy gdrive:omniforge/data/tokenized/{fname} {LOCAL_OMNIFORGE}/data/tokenized/")
    rclone(f"copy gdrive:omniforge/logs/training_log.csv {LOCAL_OMNIFORGE}/logs/")

def run_data_pipeline():
    train_bin = Path(f"{LOCAL_OMNIFORGE}/data/tokenized/train.bin")
    if train_bin.exists() and train_bin.stat().st_size > 100:
        print("\n[pipeline] train.bin exists — skipping data pipeline.")
        return
    print("\n" + "="*60)
    print("STEP 5: Running data pipeline (first time only)")
    print("="*60)
    run("python dataset_collector.py --max-docs 500000")
    run("python dataset_cleaner.py")
    run("python deduplicator.py")
    run("python prepare_dataset.py")
    for fname in ["train.bin", "val.bin", "test.bin"]:
        src = f"{LOCAL_OMNIFORGE}/data/tokenized/{fname}"
        if Path(src).exists():
            rclone(f"copy {src} gdrive:omniforge/data/tokenized/")

def run_training():
    print("\n" + "="*60)
    print("STEP 6: Training")
    print("="*60)
    r = subprocess.run("python train.py", shell=True)
    print(f"[train] Exited with code: {r.returncode}")
    return r.returncode

def save_final_state():
    print("\n" + "="*60)
    print("STEP 7: Saving final state")
    print("="*60)
    rclone(f"copy {LOCAL_OMNIFORGE}/checkpoints gdrive:omniforge/checkpoints --transfers=4")
    rclone(f"copy {LOCAL_OMNIFORGE}/logs/training_log.csv gdrive:omniforge/logs/")
    r = subprocess.run("rclone lsf gdrive:omniforge/checkpoints/ | sort", shell=True, capture_output=True, text=True)
    if r.returncode == 0:
        files = [f for f in r.stdout.strip().split("\n") if f]
        print(f"[save] {len(files)} file(s) on Drive:")
        for f in files[-5:]: print(f"[save]   {f}")

def main():
    print("="*60)
    print("  OmniForge TinyLlama Fine-tuning on Kaggle")
    print("="*60)
    install_dependencies()
    setup_rclone()
    setup_project()
    restore_from_drive()
    run_data_pipeline()
    exit_code = run_training()
    save_final_state()
    print("\n" + "="*60)
    print("Done!" if exit_code == 0 else f"Exited with code {exit_code} — progress saved.")
    print("Next session triggers automatically tomorrow.")
    print("="*60)

if __name__ == "__main__":
    main()
