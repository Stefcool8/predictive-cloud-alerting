"""
Evaluation metrics tailored for highly imbalanced predictive alerting tasks.
Focuses on Precision, Recall, F2-Score, and PR-AUC rather than standard accuracy.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

from src import config


def evaluate_alerting_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = config.ALERT_THRESHOLD) -> dict:
    """
    Evaluates model predictions at a specific alert threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # F2 explicitly weights Recall higher than Precision
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }


def calculate_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculates the Area Under the Precision-Recall Curve (PR-AUC).
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall_vals, precision_vals)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Finds the probability threshold that maximizes the F2-score.
    In SRE, catching an incident (Recall) is more important than avoiding
    a false alarm (Precision), so F2 is the superior optimization target.
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob)

    # F2 Score Formula: 5 * (P * R) / (4 * P + R)
    f2_scores = 5 * (precision_vals * recall_vals) / ((4 * precision_vals) + recall_vals + 1e-10)

    optimal_idx = np.argmax(f2_scores)

    if optimal_idx < len(thresholds):
        return thresholds[optimal_idx]
    return 0.5


def simulate_stateful_alerts(y_prob: np.ndarray, threshold: float, cooldown_steps: int = 30) -> np.ndarray:
    """
    Simulates a real-world alerting engine with cooldown periods to prevent pager storms.
    """
    alerts_fired = np.zeros_like(y_prob, dtype=int)
    in_cooldown_until = -1

    for t, prob in enumerate(y_prob):
        if prob >= threshold and t > in_cooldown_until:
            alerts_fired[t] = 1
            in_cooldown_until = t + cooldown_steps

    return alerts_fired


if __name__ == "__main__":
    print("Testing metrics with dummy data...")
    dummy_y_true = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    dummy_y_prob = np.array([0.1, 0.2, 0.05, 0.6, 0.3, 0.8, 0.4, 0.1, 0.2, 0.9])

    pr_auc_score = calculate_pr_auc(dummy_y_true, dummy_y_prob)
    metrics = evaluate_alerting_model(dummy_y_true, dummy_y_prob, threshold=0.5)
    best_thresh = find_optimal_threshold(dummy_y_true, dummy_y_prob)

    print(f"PR-AUC: {pr_auc_score:.4f}")
    print(f"Metrics at threshold=0.5: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F2={metrics['f2_score']:.2f}")
    print(f"Optimal Threshold for F2: {best_thresh:.4f}")
