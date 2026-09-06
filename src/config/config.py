from pathlib import Path


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
TRAINED_MODEL_DIR = MODELS_DIR / "trained"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"


# ==========================================================
# Dataset
# ==========================================================

DATASET_NAME = "Amazon Fine Food Reviews"

RAW_DATA_FILE = RAW_DATA_DIR / "Reviews.csv"

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "clean_reviews.csv"


# ==========================================================
# Model
# ==========================================================

MODEL_NAME = "distilbert-base-uncased"

NUM_CLASSES = 3


# ==========================================================
# Training
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

MAX_SEQUENCE_LENGTH = 256

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

EPOCHS = 3


# ==========================================================
# Training Files
# ==========================================================

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"

VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.csv"

TEST_FILE = PROCESSED_DATA_DIR / "test.csv"


# ==========================================================
# HuggingFace
# ==========================================================

TOKENIZER_NAME = MODEL_NAME


# ==========================================================
# Device Configuration
# ==========================================================

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


MODEL_SAVE_PATH = TRAINED_MODEL_DIR / "distilbert_sentiment"

DEVICE = get_device()

# Ensure directories exist
for _directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    CHECKPOINT_DIR,
    TRAINED_MODEL_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
]:
    _directory.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Labels
# ==========================================================

LABEL_MAPPING = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive"
}


LABEL_TO_ID = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}


ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}