from pathlib import Path
from typing import Dict

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config.config import (
    MAX_SEQUENCE_LENGTH,
    TOKENIZER_NAME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AmazonReviewDataset(Dataset):
    """
    Custom PyTorch Dataset for Amazon Review Sentiment Analysis.

    This dataset:
    - Reads CSV files
    - Tokenizes reviews using Hugging Face tokenizer
    - Returns tensors ready for DistilBERT
    """

    def __init__(
        self,
        csv_file: Path,
        tokenizer=None,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:

        logger.info(f"Loading dataset: {csv_file}")

        self.dataframe = pd.read_csv(csv_file)

        self.reviews = self.dataframe["review"].astype(str).tolist()

        self.labels = self.dataframe["label"].tolist()

        self.max_length = max_length

        if tokenizer is None:
            logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_NAME
            )

        else:
            self.tokenizer = tokenizer

        logger.info(
            f"Dataset Loaded Successfully ({len(self.reviews):,} samples)"
        )

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:

        review = self.reviews[index]

        label = self.labels[index]

        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        return {

            "input_ids": encoding["input_ids"].flatten(),

            "attention_mask": encoding[
                "attention_mask"
            ].flatten(),

            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


class DatasetFactory:
    """
    Helper class for creating datasets.
    """

    def __init__(self) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME
        )

    def create_dataset(
        self,
        csv_file: Path,
    ) -> AmazonReviewDataset:

        return AmazonReviewDataset(

            csv_file=csv_file,

            tokenizer=self.tokenizer,

            max_length=MAX_SEQUENCE_LENGTH,

        )