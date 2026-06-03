# OmniForge — Setup Instructions

## Quick Start (Any System)

```bash
# 1. Create directories
mkdir -p data/raw data/clean data/tokenized tokenizer checkpoints logs

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline (in order)
python dataset_collector.py --max-docs 500000
python dataset_cleaner.py
python deduplicator.py
python train_tokenizer.py
python prepare_dataset.py
python train.py
```

## Training on Google Colab

Open `colab_train.ipynb` in Google Colab and run cells sequentially.

## Inference

```bash
# CLI streaming inference
python inference.py --prompt "def fibonacci(n):"

# Web server
python server.py
# Then open http://localhost:8000 in your browser
```

## Evaluation

```bash
python evaluate.py
```

## Status Dashboard

```bash
python status.py
```

## Termux (Android) Setup

Run:
```bash
chmod +x setup_termux.sh
./setup_termux.sh
```

Then run the pipeline steps above.

## Architecture Overview

```
OmniForge-125M: Decoder-only Transformer
  - 12 layers, 12 heads, hidden dim 768
  - SwiGLU feed-forward (hidden dim 3072)
  - RoPE position encoding
  - RMSNorm pre-normalization
  - Weight tying (embedding ↔ lm_head)
  - Flash Attention (falls back to manual)
  - Context length: 2048 tokens
  - ~125 million parameters
```

## Pipeline Order

1. `dataset_collector.py` — Downloads code data from Hugging Face
2. `dataset_cleaner.py` — Filters and quality-scores documents
3. `deduplicator.py` — Removes exact duplicates (SHA256)
4. `train_tokenizer.py` — Trains BPE tokenizer (vocab size 32,000)
5. `prepare_dataset.py` — Tokenizes and packs into binary arrays
6. `train.py` — Trains the model with checkpointing and logging
7. `inference.py` — CLI/API inference with trained model
8. `server.py` — FastAPI streaming inference server
9. `evaluate.py` — Perplexity and code completion evaluation

## Configuration

All hyperparameters are in `config.py`. Edit that file to change:
- Model architecture (layers, heads, hidden dim)
- Training settings (learning rate, batch size, etc.)
- Dataset paths
- Tokenizer settings
- Inference defaults

## File Structure

```
omniforge/
├── config.py              # Central configuration
├── model.py               # Transformer model implementation
├── dataset_collector.py   # Data downloader
├── dataset_cleaner.py     # Data filter/quality scorer
├── deduplicator.py        # Duplicate remover
├── train_tokenizer.py     # BPE tokenizer training
├── prepare_dataset.py     # Tokenization and packing
├── train.py               # Training loop
├── inference.py           # Inference utilities
├── server.py              # FastAPI web server
├── evaluate.py            # Evaluation script
├── status.py              # Project status dashboard
├── setup_termux.sh        # Termux setup script
├── requirements.txt       # Python dependencies
├── colab_train.ipynb      # Google Colab notebook
├── SETUP_INSTRUCTIONS.md  # This file
├── data/                  # Dataset files (auto-created)
│   ├── raw/
│   ├── clean/
│   └── tokenized/         # train.bin, val.bin, test.bin
├── tokenizer/             # Trained tokenizer files
├── checkpoints/           # Model checkpoints
└── logs/                  # Training logs
```