#!/usr/bin/env bash
set -euo pipefail

printf '\n[OmniForge] Step 1/7: Updating Termux package index...\n'
pkg update -y

printf '\n[OmniForge] Step 2/7: Upgrading installed packages...\n'
pkg upgrade -y

printf '\n[OmniForge] Step 3/7: Installing required system packages...\n'
pkg install -y python git wget clang make cmake pkg-config libffi openssl rust

printf '\n[OmniForge] Step 4/7: Upgrading pip, setuptools, and wheel...\n'
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel

printf '\n[OmniForge] Step 5/7: Creating OmniForge project directories...\n'
mkdir -p data/raw data/clean data/tokenized tokenizer checkpoints logs

printf '\n[OmniForge] Step 6/7: Installing Python dependencies from requirements.txt...\n'
python -m pip install -r requirements.txt

printf '\n[OmniForge] Step 7/7: Verifying Python environment...\n'
python --version
python -m pip --version

printf '\n[OmniForge] SUCCESS: Termux setup completed. Project directories are ready.\n'
printf '[OmniForge] Next command: python dataset_collector.py --max-docs 10000\n'