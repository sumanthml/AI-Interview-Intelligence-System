import re
import unicodedata

import pandas as pd

from src.config.config import (
    LABEL_MAPPING,
    LABEL_TO_ID,
    PROCESSED_DATA_FILE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Handles preprocessing of the Amazon Reviews dataset.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe.copy()

    def preprocess(self) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        """

        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline...")
        logger.info("=" * 60)

        self._remove_duplicates()

        self._remove_missing_values()

        self._combine_text_columns()

        self._clean_text()

        self._generate_sentiment_labels()

        self._encode_labels()

        self._keep_required_columns()

        logger.info("=" * 60)
        logger.info("Preprocessing completed successfully.")
        logger.info(f"Final Dataset Shape : {self.dataframe.shape}")
        logger.info("=" * 60)

        return self.dataframe

    def save(self) -> None:
        """
        Save processed dataset.
        """

        PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

        self.dataframe.to_csv(PROCESSED_DATA_FILE, index=False)

        logger.info(f"Processed dataset saved to:")
        logger.info(PROCESSED_DATA_FILE)

    def _remove_duplicates(self) -> None:

        logger.info("Removing duplicate reviews...")

        before = len(self.dataframe)

        self.dataframe.drop_duplicates(
            subset=["Text"],
            inplace=True,
            ignore_index=True,
        )

        after = len(self.dataframe)

        logger.info(f"Removed {before-after:,} duplicate reviews.")

    def _remove_missing_values(self) -> None:

        logger.info("Removing missing values...")

        before = len(self.dataframe)

        self.dataframe.dropna(
            subset=["Score", "Summary", "Text"],
            inplace=True,
        )

        after = len(self.dataframe)

        logger.info(f"Removed {before-after:,} rows with missing values.")

    def _combine_text_columns(self) -> None:

        logger.info("Combining Summary and Text...")

        self.dataframe["review"] = (
            self.dataframe["Summary"].astype(str)
            + ". "
            + self.dataframe["Text"].astype(str)
        )

        logger.info("Review column created.")

    def _clean_text(self) -> None:

        logger.info("Cleaning review text...")

        review = self.dataframe["review"].astype(str)

        review = review.str.replace(
            r"<.*?>",
            "",
            regex=True,
        )

        review = review.str.replace(
            r"http\S+|www\S+",
            "",
            regex=True,
        )

        review = review.str.replace(
            r"\s+",
            " ",
            regex=True,
        )

        review = review.str.strip()

        review = review.apply(
            lambda text: unicodedata.normalize("NFKC", text)
        )

        self.dataframe["review"] = review

        logger.info("Text cleaning completed.")

    def _generate_sentiment_labels(self) -> None:

        logger.info("Generating sentiment labels...")

        self.dataframe["sentiment"] = (
            self.dataframe["Score"]
            .map(LABEL_MAPPING)
        )

        logger.info("Sentiment labels created.")

    def _encode_labels(self) -> None:

        logger.info("Encoding labels...")

        self.dataframe["label"] = (
            self.dataframe["sentiment"]
            .map(LABEL_TO_ID)
        )

        logger.info("Label encoding completed.")

    def _keep_required_columns(self) -> None:

        logger.info("Selecting final columns...")

        self.dataframe = self.dataframe[
            [
                "review",
                "Score",
                "sentiment",
                "label",
            ]
        ]

        logger.info("Final columns selected.")