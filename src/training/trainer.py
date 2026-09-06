import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    METRICS_DIR,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    TOKENIZER_NAME,
    TRAIN_BATCH_SIZE,
    VALID_BATCH_SIZE,
    WEIGHT_DECAY,
)
from src.evaluation.metrics import compute_classification_metrics
from src.training.dataset import AmazonReviewDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DistilBertTrainer:
    """
    Handles fine-tuning of DistilBERT for 3-class sentiment classification.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        epochs: int = EPOCHS,
        train_batch_size: int = TRAIN_BATCH_SIZE,
        valid_batch_size: int = VALID_BATCH_SIZE,
        device: str = DEVICE,
        model_save_path: Path = MODEL_SAVE_PATH,
    ) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.device = torch.device(device)
        self.model_save_path = model_save_path

        logger.info(f"Using Compute Device: {self.device}")

        # Initialize Tokenizer and Model
        logger.info(f"Loading Base Model and Tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
        )
        self.model.to(self.device)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1_weighted": [],
        }

    def train(
        self,
        train_csv: Path,
        val_csv: Path,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute full training loop.
        """
        logger.info("Initializing datasets and data loaders...")
        train_dataset = AmazonReviewDataset(train_csv, tokenizer=self.tokenizer)
        val_dataset = AmazonReviewDataset(val_csv, tokenizer=self.tokenizer)

        if max_train_samples and max_train_samples < len(train_dataset):
            indices = list(range(max_train_samples))
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
            logger.info(f"Subsampled train set to {max_train_samples:,} samples for fast training.")

        if max_val_samples and max_val_samples < len(val_dataset):
            indices = list(range(max_val_samples))
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
            logger.info(f"Subsampled val set to {max_val_samples:,} samples for fast training.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
        )

        total_steps = len(train_loader) * self.epochs
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        best_val_f1 = 0.0
        start_time = time.time()

        logger.info("=" * 60)
        logger.info(f"Starting Training: {self.epochs} Epochs | Total Steps: {total_steps}")
        logger.info("=" * 60)

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            self.model.train()
            total_train_loss = 0.0

            for step, batch in enumerate(train_loader, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                if step % max(1, len(train_loader) // 5) == 0 or step == len(train_loader):
                    logger.info(
                        f"Epoch {epoch}/{self.epochs} | Step {step}/{len(train_loader)} | "
                        f"Batch Loss: {loss.item():.4f}"
                    )

            avg_train_loss = total_train_loss / len(train_loader)
            self.history["train_loss"].append(avg_train_loss)

            # Validation
            val_metrics = self.evaluate(val_loader)
            avg_val_loss = val_metrics["val_loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_weighted"]

            self.history["val_loss"].append(avg_val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_f1_weighted"].append(val_f1)

            epoch_time = time.time() - epoch_start
            logger.info("-" * 60)
            logger.info(
                f"Epoch {epoch} Complete in {epoch_time:.1f}s | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )
            logger.info("-" * 60)

            # Save best model checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model()
                logger.info(f"New best model saved with Val F1: {best_val_f1:.4f}")

        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time/60:.2f} minutes.")

        # Save training history
        history_path = METRICS_DIR / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"Training history saved to {history_path}")

        return self.history

    def evaluate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Run validation loop and calculate loss and metrics."""
        self.model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                total_val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        metrics = compute_classification_metrics(np.array(all_labels), np.array(all_preds))
        metrics["val_loss"] = avg_val_loss
        return metrics

    def save_model(self, save_path: Optional[Path] = None) -> None:
        """Save model and tokenizer to disk."""
        target_path = save_path or self.model_save_path
        target_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target_path))
        self.tokenizer.save_pretrained(str(target_path))
        logger.info(f"Model and tokenizer successfully saved to: {target_path}")
