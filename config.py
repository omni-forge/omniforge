"""
OmniForge central configuration.

Every project component imports settings from this file. No training,
dataset, tokenizer, inference, or path constants should be hardcoded in
other modules.

Source: ZIP1 (primary, dataclass-based architecture) with ZIP6 improvements
        (bfloat16 detection, enhanced error handling paths)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# -----------------------------
# Path settings
# -----------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
CLEAN_DATA_DIR: Path = DATA_DIR / "clean"
TOKENIZED_DATA_DIR: Path = DATA_DIR / "tokenized"
TOKENIZER_DIR: Path = PROJECT_ROOT / "tokenizer"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
LOG_DIR: Path = PROJECT_ROOT / "logs"

RAW_DATASET_PATH: Path = RAW_DATA_DIR / "raw_dataset.jsonl.gz"
CLEAN_DATASET_PATH: Path = CLEAN_DATA_DIR / "clean_dataset.jsonl.gz"
DEDUPED_DATASET_PATH: Path = CLEAN_DATA_DIR / "deduped_dataset.jsonl.gz"
TRAIN_BIN_PATH: Path = TOKENIZED_DATA_DIR / "train.bin"
VAL_BIN_PATH: Path = TOKENIZED_DATA_DIR / "val.bin"
TEST_BIN_PATH: Path = TOKENIZED_DATA_DIR / "test.bin"
TRAINING_LOG_PATH: Path = LOG_DIR / "training_log.csv"

DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    TOKENIZED_DATA_DIR,
    TOKENIZER_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
]


# -----------------------------
# Model architecture settings
# -----------------------------
MODEL_NAME: str = "OmniForge-125M"
MODEL_TYPE: str = "decoder-only-transformer"
N_LAYERS: int = 12
N_HEADS: int = 12
HIDDEN_DIM: int = 768
FFN_DIM: int = 3072
CONTEXT_LENGTH: int = 2048
VOCAB_SIZE: int = 32000
DROPOUT: float = 0.0
RMS_NORM_EPS: float = 1e-5
ROPE_BASE: float = 10000.0
USE_FLASH_ATTENTION: bool = True


# -----------------------------
# Special tokens
# -----------------------------
PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
BOS_TOKEN: str = "<BOS>"
EOS_TOKEN: str = "<EOS>"
EOD_TOKEN: str = "<EOD>"

PAD_TOKEN_ID: int = 0
UNK_TOKEN_ID: int = 1
BOS_TOKEN_ID: int = 2
EOS_TOKEN_ID: int = 3
EOD_TOKEN_ID: int = 4

SPECIAL_TOKENS: List[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, EOD_TOKEN]
SPECIAL_TOKEN_IDS: Dict[str, int] = {
    PAD_TOKEN: PAD_TOKEN_ID,
    UNK_TOKEN: UNK_TOKEN_ID,
    BOS_TOKEN: BOS_TOKEN_ID,
    EOS_TOKEN: EOS_TOKEN_ID,
    EOD_TOKEN: EOD_TOKEN_ID,
}


# -----------------------------
# Training settings
# -----------------------------
BATCH_SIZE: int = 8
GRADIENT_ACCUMULATION_STEPS: int = 8
LEARNING_RATE: float = 3e-4
MIN_LEARNING_RATE: float = 1e-5
WARMUP_STEPS: int = 2000
WEIGHT_DECAY: float = 0.1
BETA1: float = 0.9
BETA2: float = 0.95
MAX_STEPS: int = 100000
EVAL_INTERVAL: int = 500
SAVE_INTERVAL: int = 1000
GRAD_CLIP: float = 1.0
SEED: int = 1337
NUM_WORKERS: int = 2
DTYPE: str = "float16"


# -----------------------------
# Dataset settings
# -----------------------------
DATASET_SOURCE: str = "bigcode/the-stack-dedup"
DATASET_LANGUAGES: List[str] = ["python", "javascript"]
TRAIN_SPLIT: float = 0.95
VAL_SPLIT: float = 0.04
TEST_SPLIT: float = 0.01
SPLIT_RATIOS: Tuple[float, float, float] = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
MAX_RAW_DOCUMENT_CHARS: int = 100000
MIN_RAW_DOCUMENT_CHARS: int = 100
MAX_TOKENIZER_DOCS: int = 500000
COLLECTOR_LOG_INTERVAL: int = 10000
COLLECTOR_MAX_RETRIES: int = 5
COLLECTOR_INITIAL_BACKOFF_SECONDS: float = 2.0


# -----------------------------
# Tokenizer settings
# -----------------------------
TOKENIZER_TYPE: str = "BPE"
TOKENIZER_VOCAB_SIZE: int = VOCAB_SIZE
TOKENIZER_MIN_FREQUENCY: int = 2
TOKENIZER_OUTPUT_DIR: Path = TOKENIZER_DIR


# -----------------------------
# Inference settings
# -----------------------------
DEFAULT_TEMPERATURE: float = 0.8
DEFAULT_TOP_K: int = 50
DEFAULT_TOP_P: float = 0.95
DEFAULT_MAX_NEW_TOKENS: int = 256
DEFAULT_DO_SAMPLE: bool = True


# -----------------------------
# Server settings
# -----------------------------
SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 8000


def ensure_directories() -> None:
    """Create all configured project directories."""
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ArchitectureConfig:
    vocab_size: int = VOCAB_SIZE
    context_length: int = CONTEXT_LENGTH
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    hidden_dim: int = HIDDEN_DIM
    ffn_dim: int = FFN_DIM
    dropout: float = DROPOUT
    rms_norm_eps: float = RMS_NORM_EPS
    rope_base: float = ROPE_BASE
    use_flash_attention: bool = USE_FLASH_ATTENTION
    pad_token_id: int = PAD_TOKEN_ID
    bos_token_id: int = BOS_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID
    eod_token_id: int = EOD_TOKEN_ID


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = BATCH_SIZE
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = LEARNING_RATE
    min_learning_rate: float = MIN_LEARNING_RATE
    warmup_steps: int = WARMUP_STEPS
    weight_decay: float = WEIGHT_DECAY
    beta1: float = BETA1
    beta2: float = BETA2
    max_steps: int = MAX_STEPS
    eval_interval: int = EVAL_INTERVAL
    save_interval: int = SAVE_INTERVAL
    grad_clip: float = GRAD_CLIP
    seed: int = SEED


@dataclass(frozen=True)
class DatasetConfig:
    source: str = DATASET_SOURCE
    languages: List[str] = field(default_factory=lambda: list(DATASET_LANGUAGES))
    split_ratios: Tuple[float, float, float] = SPLIT_RATIOS
    raw_dataset_path: Path = RAW_DATASET_PATH
    clean_dataset_path: Path = CLEAN_DATASET_PATH
    deduped_dataset_path: Path = DEDUPED_DATASET_PATH
    train_bin_path: Path = TRAIN_BIN_PATH
    val_bin_path: Path = VAL_BIN_PATH
    test_bin_path: Path = TEST_BIN_PATH


@dataclass(frozen=True)
class InferenceConfig:
    default_temperature: float = DEFAULT_TEMPERATURE
    default_top_k: int = DEFAULT_TOP_K
    default_top_p: float = DEFAULT_TOP_P
    default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    default_do_sample: bool = DEFAULT_DO_SAMPLE


# Aliases for model.py compatibility
N_LAYER = N_LAYERS
N_HEAD = N_HEADS
N_EMBD = HIDDEN_DIM
N_KV_HEAD = N_HEADS
BLOCK_SIZE = CONTEXT_LENGTH