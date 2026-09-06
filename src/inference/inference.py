from typing import Any, Dict, List

from src.inference.predictor import SentimentPredictor


def predict_sentiment(review_text: str) -> Dict[str, Any]:
    """Helper function to predict single review sentiment."""
    predictor = SentimentPredictor()
    return predictor.predict(review_text)


def batch_predict_sentiment(reviews: List[str]) -> List[Dict[str, Any]]:
    """Helper function to predict sentiment for a batch of reviews."""
    predictor = SentimentPredictor()
    return predictor.predict_batch(reviews)


if __name__ == "__main__":
    sample = "The build quality is stellar and battery life exceeded my expectations!"
    result = predict_sentiment(sample)
    print("Inference Result:", result)
