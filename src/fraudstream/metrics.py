"""Fraud-detection metrics. Accuracy is useless at a ~1% fraud rate, so we use ranking and
alert-level metrics instead."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """The score threshold with the best F1 (balance of missed fraud and false alerts)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))]) if len(thresholds) else 0.5


def alert_metrics(y_true: np.ndarray, alerts: np.ndarray) -> dict[str, float]:
    y_true, alerts = np.asarray(y_true, bool), np.asarray(alerts, bool)
    caught = int((y_true & alerts).sum())
    precision = caught / alerts.sum() if alerts.sum() else 0.0
    recall = caught / y_true.sum() if y_true.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alerts": int(alerts.sum()),
        "caught": caught,
        "frauds": int(y_true.sum()),
    }


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Ranking quality (PR-AUC, ROC-AUC) plus the alert metrics at `threshold`."""
    y_true = np.asarray(y_true)
    both_classes = 0 < y_true.sum() < len(y_true)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else float("nan"),
        **alert_metrics(y_true, np.asarray(scores) >= threshold),
    }
