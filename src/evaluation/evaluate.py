import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.config import (
    DEVICE,
    ID_TO_LABEL,
    METRICS_DIR,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    TEST_FILE,
    TOKENIZER_NAME,
    VALID_BATCH_SIZE,
)
from src.evaluation.metrics import compute_classification_metrics
from src.training.dataset import AmazonReviewDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates a trained Transformer model against test dataset.
    """

    def __init__(
        self,
        model_path: Path = MODEL_SAVE_PATH,
        device: str = DEVICE,
        batch_size: int = VALID_BATCH_SIZE,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None

    def _load_model(self) -> None:
        if self.model is None:
            logger.info(f"Loading evaluation model from: {self.model_path}")
            if self.model_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            else:
                logger.warning(f"Trained model not found at {self.model_path}. Loading baseline {TOKENIZER_NAME}")
                self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    TOKENIZER_NAME,
                    num_labels=NUM_CLASSES,
                )
            self.model.to(self.device)
            self.model.eval()

    def evaluate(
        self,
        test_file: Path = TEST_FILE,
        save_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Run model evaluation on test dataset.
        """
        self._load_model()
        logger.info(f"Evaluating on {test_file}...")

        dataset = AmazonReviewDataset(csv_file=test_file, tokenizer=self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].numpy()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        results = compute_classification_metrics(y_true, y_pred)
        logger.info(f"Evaluation Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Evaluation Weighted F1: {results['f1_weighted']:.4f}")

        if save_metrics:
            metrics_file = METRICS_DIR / "evaluation_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved evaluation metrics to: {metrics_file}")

        return results


def main() -> None:
    evaluator = ModelEvaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
