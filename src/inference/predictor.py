import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.config import (
    DEVICE,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MAX_SEQUENCE_LENGTH,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    TOKENIZER_NAME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentPredictor:
    """
    Production-grade sentiment inference engine for Amazon Reviews.
    Supports single review, batch reviews, and token-level importance scoring.
    """

    FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = DEVICE,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        self.device = torch.device(device)
        self.max_length = max_length
        self.model_path = Path(model_path) if model_path else MODEL_SAVE_PATH
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load fine-tuned model from checkpoint or fallback to pre-trained pipeline."""
        if self.model_path.exists() and (self.model_path / "config.json").exists():
            try:
                logger.info(f"Loading fine-tuned model from {self.model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
                self.model.to(self.device)
                self.model.eval()
                logger.info("Fine-tuned model loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Could not load local model: {e}. Falling back to pre-trained model.")

        # Fallback to standard pre-trained HuggingFace model
        try:
            logger.info(f"Loading pre-trained model ({self.FALLBACK_MODEL})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.FALLBACK_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.FALLBACK_MODEL)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Fallback sentiment model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load fallback model: {e}. Initializing base {MODEL_NAME}.")
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=NUM_CLASSES,
            )
            self.model.to(self.device)
            self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single review string.
        """
        start_time = time.time()
        if not text or not text.strip():
            return {
                "sentiment": "Neutral",
                "label_id": 1,
                "confidence": 0.33,
                "probabilities": {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33},
                "latency_ms": 0.0,
            }

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()

        if self.model.config.num_labels == 2:
            # Binary classification fallback (SST-2: 0->Negative, 1->Positive)
            neg_p, pos_p = probs[0], probs[1]
            diff = abs(pos_p - neg_p)
            if diff < 0.25:
                neu_p = 0.5 + (0.25 - diff)
                neg_p = (1.0 - neu_p) * neg_p / (neg_p + pos_p)
                pos_p = (1.0 - neu_p) * pos_p / (neg_p + pos_p)
            else:
                neu_p = 0.05
                neg_p = (1.0 - neu_p) * neg_p
                pos_p = (1.0 - neu_p) * pos_p
            
            prob_dict = {
                "Negative": round(neg_p, 4),
                "Neutral": round(neu_p, 4),
                "Positive": round(pos_p, 4),
            }
        else:
            # 3-class classification
            prob_dict = {
                "Negative": round(probs[0], 4),
                "Neutral": round(probs[1], 4),
                "Positive": round(probs[2], 4),
            }

        sentiment = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[sentiment]
        label_id = LABEL_TO_ID.get(sentiment, 1)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "sentiment": sentiment,
            "label_id": label_id,
            "confidence": confidence,
            "probabilities": prob_dict,
            "latency_ms": latency_ms,
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Batch prediction for a list of reviews.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for item in chunk:
                results.append(self.predict(str(item)))
        return results

    def explain(self, text: str) -> Dict[str, Any]:
        """
        Token-level attribution and importance analysis for explainability.
        Computes token gradients and attributions to identify words that impacted sentiment.
        """
        if not text or not text.strip():
            return {"tokens": [], "attributions": [], "predicted_sentiment": "Neutral"}

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        input_ids = inputs["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.retain_grad()

        outputs = self.model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()

        pred_idx = int(torch.argmax(logits, dim=1).item())
        target_score = logits[0, pred_idx]
        self.model.zero_grad()
        target_score.backward()

        grads = embeddings.grad[0].cpu().numpy()
        token_attributions = (grads * embeddings[0].detach().cpu().numpy()).sum(axis=1)

        # Normalize attributions between -1.0 and 1.0
        max_abs = max(abs(token_attributions.max()), abs(token_attributions.min()), 1e-6)
        normalized_attributions = (token_attributions / max_abs).tolist()

        # Clean special tokens
        clean_tokens = []
        clean_scores = []
        for t, s in zip(tokens, normalized_attributions):
            if t in ["[CLS]", "[SEP]", "<s>", "</s>", "<pad>"]:
                continue
            clean_tokens.append(t.replace("##", ""))
            clean_scores.append(round(s, 4))

        pred_sentiment = "Positive" if pred_idx == 2 or (self.model.config.num_labels == 2 and pred_idx == 1) else ("Negative" if pred_idx == 0 else "Neutral")

        return {
            "tokens": clean_tokens,
            "attributions": clean_scores,
            "predicted_sentiment": pred_sentiment,
            "full_prediction": self.predict(text),
        }
