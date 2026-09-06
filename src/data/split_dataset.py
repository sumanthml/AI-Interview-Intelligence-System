from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetSplitter:
    """
    Split the processed dataset into train, validation,
    and test sets using stratified sampling.
    """

    def __init__(self, file_path: Path = PROCESSED_DATA_FILE) -> None:
        self.file_path = file_path

    def split(self) -> None:
        """
        Perform train-validation-test split and save CSV files.
        """

        logger.info("Loading processed dataset...")

        dataframe = pd.read_csv(self.file_path)

        logger.info(f"Dataset Size : {len(dataframe):,}")

        train_df, temp_df = train_test_split(
            dataframe,
            test_size=TEST_SIZE + VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=dataframe["label"],
        )

        validation_ratio = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)

        validation_df, test_df = train_test_split(
            temp_df,
            test_size=1 - validation_ratio,
            random_state=RANDOM_STATE,
            stratify=temp_df["label"],
        )

        train_path = PROCESSED_DATA_DIR / "train.csv"
        validation_path = PROCESSED_DATA_DIR / "validation.csv"
        test_path = PROCESSED_DATA_DIR / "test.csv"

        train_df.to_csv(train_path, index=False)
        validation_df.to_csv(validation_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Training Samples   : {len(train_df):,}")
        logger.info(f"Validation Samples: {len(validation_df):,}")
        logger.info(f"Testing Samples   : {len(test_df):,}")

        logger.info("Dataset splitting completed.")