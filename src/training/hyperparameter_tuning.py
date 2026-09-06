from pathlib import Path
from typing import Any, Dict

import optuna
import torch

from src.config.config import (
    MODEL_NAME,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    TRAIN_FILE,
    VALIDATION_FILE,
)
from src.training.trainer import DistilBertTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for tuning hyperparameters.
    """
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("train_batch_size", [16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)

    trainer = DistilBertTrainer(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        learning_rate=lr,
        weight_decay=weight_decay,
        epochs=1,
        train_batch_size=batch_size,
        valid_batch_size=batch_size,
    )

    history = trainer.train(
        train_csv=TRAIN_FILE,
        val_csv=VALIDATION_FILE,
        max_train_samples=500,
        max_val_samples=200,
    )

    val_f1 = history["val_f1_weighted"][-1] if history["val_f1_weighted"] else 0.0
    return float(val_f1)


def tune_hyperparameters(n_trials: int = 5) -> Dict[str, Any]:
    """
    Run Optuna study to optimize hyperparameters.
    """
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="maximize", study_name="distilbert_amazon_sentiment")
    logger.info(f"Starting Optuna Hyperparameter Study ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best Trial Value (Val F1): {study.best_value:.4f}")
    logger.info(f"Best Parameters: {study.best_params}")
    return study.best_params


if __name__ == "__main__":
    if TRAIN_FILE.exists() and VALIDATION_FILE.exists():
        tune_hyperparameters(n_trials=3)
    else:
        print("Please prepare train.csv and validation.csv first.")
