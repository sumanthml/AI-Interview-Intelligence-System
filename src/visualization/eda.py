import pandas as pd

from src.config.config import PROCESSED_DATA_FILE
from src.utils.logger import get_logger
from src.visualization.plots import PlotGenerator

logger = get_logger(__name__)


class ExploratoryDataAnalysis:
    """
    Performs Exploratory Data Analysis (EDA) on the processed dataset.
    """

    def __init__(self) -> None:
        self.dataframe = pd.read_csv(PROCESSED_DATA_FILE)
        self.plotter = PlotGenerator()

    def run(self) -> None:
        """
        Execute complete EDA pipeline.
        """

        logger.info("=" * 60)
        logger.info("Starting Exploratory Data Analysis")
        logger.info("=" * 60)

        self._dataset_overview()

        self._missing_values()

        self._class_distribution()

        self._review_statistics()

        self._generate_plots()

        logger.info("=" * 60)
        logger.info("EDA Completed Successfully")
        logger.info("=" * 60)

    def _dataset_overview(self) -> None:

        logger.info(f"Dataset Shape : {self.dataframe.shape}")

        logger.info(f"Columns : {list(self.dataframe.columns)}")

    def _missing_values(self) -> None:

        logger.info("Missing Values")

        logger.info(self.dataframe.isnull().sum())

    def _class_distribution(self) -> None:

        logger.info("Sentiment Distribution")

        logger.info(self.dataframe["sentiment"].value_counts())

    def _review_statistics(self) -> None:

        self.dataframe["review_length"] = (
            self.dataframe["review"]
            .astype(str)
            .str.len()
        )

        logger.info("Review Length Statistics")

        logger.info(self.dataframe["review_length"].describe())

    def _generate_plots(self) -> None:

        logger.info("Generating Visualizations...")

        self.plotter.sentiment_distribution(self.dataframe)

        self.plotter.rating_distribution(self.dataframe)

        self.plotter.review_length_distribution(self.dataframe)

        self.plotter.positive_wordcloud(self.dataframe)

        self.plotter.negative_wordcloud(self.dataframe)

        logger.info("All figures saved successfully.")