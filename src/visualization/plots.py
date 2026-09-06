from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from src.config.config import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PlotGenerator:
    """
    Generates exploratory data analysis plots and saves them to outputs/figures.
    Also provides helpers for Streamlit visualization.
    """

    def __init__(self, output_dir: Path = FIGURES_DIR) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use clean modern styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def sentiment_distribution(self, df: pd.DataFrame, save_filename: str = "sentiment_distribution.png") -> Path:
        """Plot and save sentiment distribution bar chart."""
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = df["sentiment"].value_counts()
        colors = ["#2ecc71" if s == "Positive" else "#e74c3c" if s == "Negative" else "#f39c12" for s in counts.index]
        
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", alpha=0.85)
        ax.set_title("Sentiment Class Distribution", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Sentiment", fontsize=12)
        ax.set_ylabel("Number of Reviews", fontsize=12)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:,}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved sentiment distribution plot to {save_path}")
        return save_path

    def rating_distribution(self, df: pd.DataFrame, save_filename: str = "rating_distribution.png") -> Path:
        """Plot and save 1-5 star rating distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))
        if "Score" in df.columns:
            counts = df["Score"].value_counts().sort_index()
            bars = ax.bar(counts.index.astype(str), counts.values, color="#3498db", edgecolor="black", alpha=0.85)
            ax.set_title("Amazon Review Star Ratings (1-5)", fontsize=14, fontweight="bold", pad=12)
            ax.set_xlabel("Rating (Stars)", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:,}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved rating distribution plot to {save_path}")
        return save_path

    def review_length_distribution(self, df: pd.DataFrame, save_filename: str = "review_length_distribution.png") -> Path:
        """Plot and save character length distribution histogram."""
        fig, ax = plt.subplots(figsize=(9, 5))
        lengths = df["review"].astype(str).str.len()
        # Clip top 1% for cleaner visualization
        clipped = lengths[lengths <= lengths.quantile(0.99)]

        ax.hist(clipped, bins=50, color="#9b59b6", edgecolor="black", alpha=0.75)
        ax.set_title("Review Character Length Distribution (99th Percentile)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Character Count", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.axvline(lengths.median(), color="red", linestyle="--", linewidth=1.5, label=f"Median: {int(lengths.median())} chars")
        ax.legend()

        plt.tight_layout()
        save_path = self.output_dir / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved review length distribution plot to {save_path}")
        return save_path

    def positive_wordcloud(self, df: pd.DataFrame, save_filename: str = "positive_wordcloud.png") -> Optional[Path]:
        """Generate and save WordCloud for Positive reviews."""
        pos_df = df[df["sentiment"] == "Positive"]
        if pos_df.empty:
            return None
        text = " ".join(pos_df["review"].astype(str).head(5000).tolist())
        wc = WordCloud(
            width=800, height=400, background_color="white",
            colormap="Greens", max_words=100, contour_width=1, contour_color="seagreen"
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Most Frequent Positive Review Words", fontsize=14, fontweight="bold", pad=12)

        plt.tight_layout()
        save_path = self.output_dir / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved positive wordcloud to {save_path}")
        return save_path

    def negative_wordcloud(self, df: pd.DataFrame, save_filename: str = "negative_wordcloud.png") -> Optional[Path]:
        """Generate and save WordCloud for Negative reviews."""
        neg_df = df[df["sentiment"] == "Negative"]
        if neg_df.empty:
            return None
        text = " ".join(neg_df["review"].astype(str).head(5000).tolist())
        wc = WordCloud(
            width=800, height=400, background_color="white",
            colormap="Reds", max_words=100, contour_width=1, contour_color="darkred"
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Most Frequent Negative Review Words", fontsize=14, fontweight="bold", pad=12)

        plt.tight_layout()
        save_path = self.output_dir / save_filename
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved negative wordcloud to {save_path}")
        return save_path
