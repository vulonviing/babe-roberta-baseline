"""Central config: paths, hyperparameters, constants. Edit here, not in notebooks."""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --- Dataset ---
HF_DATASET = "mediabiasgroup/BABE"  # public dataset card on HuggingFace
TEXT_COL = "text"
LABEL_COL = "label"  # 0 = non-biased, 1 = biased
LABEL_NAMES = ["non-biased", "biased"]

# --- Model ---
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128

# --- Training ---
SEED = 42
N_FOLDS = 5
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# --- Logging ---
WANDB_PROJECT = "babe-baseline"
WANDB_ENABLED = False  # set True after `wandb login`
