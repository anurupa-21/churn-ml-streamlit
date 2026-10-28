"""Evaluation helpers shared by the training script and the Streamlit app."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def positive_class_scores(model, X):
    """Probability (or decision score) of the positive class, used for AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """All six metrics required by the assignment for a binary problem."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")  # AUC is undefined when only one class is present
    else:
        auc = roc_auc_score(y_true, y_score)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
