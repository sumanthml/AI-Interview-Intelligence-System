from pathlib import Path

import pandas as pd

from src.config.config import RAW_DATA_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Handles loading and validating the Amazon Reviews dataset.
    """

    REQUIRED_COLUMNS = [
        "Score",
        "Summary",
        "Text"
    ]

    def __init__(self, file_path: Path = RAW_DATA_FILE) -> None:
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError
            ValueError
        """

        if not self.file_path.exists():
            logger.error(f"Dataset not found: {self.file_path}")
            raise FileNotFoundError(
                f"Dataset not found at {self.file_path}"
            )

        logger.info("Loading dataset...")

        dataframe = pd.read_csv(self.file_path)

        self._validate_columns(dataframe)

        logger.info(
            f"Dataset loaded successfully "
            f"({len(dataframe):,} rows)"
        )

        return dataframe

    def _validate_columns(self, dataframe: pd.DataFrame) -> None:
        """
        Validate required columns.
        """

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )