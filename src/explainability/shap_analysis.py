from typing import Any, Dict, List, Tuple
import html

from src.inference.predictor import SentimentPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """
    Explainability engine for Transformer sentiment models.
    Produces token attributions, word-level impact scores, and HTML color highlights.
    """

    def __init__(self) -> None:
        self.predictor = SentimentPredictor()

    def explain_text(self, text: str) -> Dict[str, Any]:
        """
        Compute token importance scores and generate HTML visualization.
        """
        explanation = self.predictor.explain(text)
        tokens = explanation["tokens"]
        attributions = explanation["attributions"]
        sentiment = explanation["predicted_sentiment"]

        html_tokens = []
        for token, score in zip(tokens, attributions):
            safe_token = html.escape(token)
            # Normalize opacity
            alpha = min(abs(score), 1.0)
            if score > 0.05:
                # Positive contribution (green)
                bg = f"rgba(46, 204, 113, {0.2 + 0.6 * alpha:.2f})"
                color = "#145a32"
            elif score < -0.05:
                # Negative contribution (red)
                bg = f"rgba(231, 76, 60, {0.2 + 0.6 * alpha:.2f})"
                color = "#78281f"
            else:
                bg = "rgba(189, 195, 199, 0.2)"
                color = "#2c3e50"

            html_tokens.append(
                f'<span style="background-color: {bg}; color: {color}; padding: 2px 5px; '
                f'margin: 2px; border-radius: 4px; display: inline-block; font-weight: 500;" '
                f'title="Attribution Score: {score:+.3f}">{safe_token}</span>'
            )

        html_markup = f"""
        <div style="line-height: 2.2; font-size: 16px; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; background: #fafafa;">
            {" ".join(html_tokens)}
        </div>
        """

        # Sort words by absolute influence
        ranked_words: List[Tuple[str, float]] = sorted(
            zip(tokens, attributions),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {
            "tokens": tokens,
            "attributions": attributions,
            "predicted_sentiment": sentiment,
            "html_visualization": html_markup,
            "top_influential_tokens": ranked_words[:10],
            "full_prediction": explanation.get("full_prediction", {}),
        }
