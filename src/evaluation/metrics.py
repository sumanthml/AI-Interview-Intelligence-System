from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config.config import ID_TO_LABEL


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute key classification metrics.

    Args:
        y_true: Ground truth labels (integer IDs)
        y_pred: Predicted labels (integer IDs)

    Returns:
        Dictionary of metrics including accuracy, precision, recall, f1, and confusion matrix.
    """
    target_names = [ID_TO_LABEL.get(i, str(i)) for i in sorted(list(set(y_true) | set(y_pred)))]
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "precision_macro": float(precision_macro),
        "recall_weighted": float(recall_weighted),
        "recall_macro": float(recall_macro),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "target_names": target_names,
    }
