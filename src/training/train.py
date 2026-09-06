import argparse
from pathlib import Path

from src.config.config import (
    EPOCHS,
    LEARNING_RATE,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_FILE,
    VALID_BATCH_SIZE,
    VALIDATION_FILE,
    WEIGHT_DECAY,
)
from src.training.trainer import DistilBertTrainer
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DistilBERT Sentiment Classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=TRAIN_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--valid_batch_size", type=int, default=VALID_BATCH_SIZE, help="Validation batch size")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples for fast training")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    if not TRAIN_FILE.exists() or not VALIDATION_FILE.exists():
        logger.error(f"Training data not found at {TRAIN_FILE} or {VALIDATION_FILE}. Run main.py first.")
        raise FileNotFoundError("Missing processed train/validation datasets.")

    trainer = DistilBertTrainer(
        model_name=MODEL_NAME,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        model_save_path=MODEL_SAVE_PATH,
    )

    trainer.train(
        train_csv=TRAIN_FILE,
        val_csv=VALIDATION_FILE,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()
