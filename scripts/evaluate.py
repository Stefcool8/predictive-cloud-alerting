"""
Unified evaluation and explainability script.
Loads the appropriate trained model, runs inference, and generates a comprehensive
terminal report proving business value, explaining threshold decisions, and
demonstrating real-world alerting adaptations.

Usage:
$ python -m scripts.evaluate --model rf
$ python -m scripts.evaluate --model hybrid
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from src import config
from src.models.registry import get_model
from src.utils.metrics import (
    calculate_pr_auc,
    simulate_stateful_alerts,
    evaluate_alerting_model,
    find_optimal_threshold,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained alerting model.")
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'hybrid'],
                        help="The model architecture to evaluate.")
    args = parser.parse_args()

    print(f"\n============================================================")
    print(f"  PREDICTIVE ALERTING EVALUATION REPORT: {args.model.upper()}")
    print(f"============================================================")


    print("\n[1] Loading Test Data & Model...")
    x = np.load(config.PROCESSED_X_PATH)
    y = np.load(config.PROCESSED_Y_PATH)

    # Chronological Split (Strictly predicting the future)
    split_index = int(len(x) * (1 - config.TEST_SIZE))
    x_test, y_test = x[split_index:], y[split_index:]

    # The baseline heuristic always checks the final timestep of the 3D window
    recent_metrics = x_test[:, -1, :]

    model = get_model(args.model)
    model.load_model(config.MODEL_PATHS[args.model])


    print("\n[2] Generating ML Predictions & Analyzing Thresholds...")
    ml_prob = model.predict_proba(x_test)
    ml_pr_auc = calculate_pr_auc(y_test, ml_prob)

    # Evaluate at Default Threshold
    ml_default_metrics = evaluate_alerting_model(y_test, ml_prob, threshold=config.ALERT_THRESHOLD)

    # Evaluate at Mathematically Optimal Threshold (Optimizing for F2)
    optimal_thresh = find_optimal_threshold(y_test, ml_prob)
    ml_optimal_metrics = evaluate_alerting_model(y_test, ml_prob, threshold=optimal_thresh)


    print("\n[3] Baseline Comparison (Proving ML Value)...")
    # A traditional DevOps alert: "Alert if any metric in the current minute > 85%"
    heuristic_prob = np.max(recent_metrics, axis=1) / 100.0
    heuristic_pr_auc = calculate_pr_auc(y_test, heuristic_prob)
    heuristic_metrics = evaluate_alerting_model(y_test, heuristic_prob, threshold=0.85)

    print("\n--- Traditional DevOps Heuristic (Current Minute > 85%) ---")
    print(f"PR-AUC:   {heuristic_pr_auc:.4f}")
    print(f"F1-Score: {heuristic_metrics['f1_score']:.4f}")
    print(f"F2-Score: {heuristic_metrics['f2_score']:.4f} (SRE Metric: Favors Recall)")
    print(f"Confusion Matrix:")
    print(f"  Caught: {heuristic_metrics['true_positives']} | Missed: {heuristic_metrics['false_negatives']} | False Alarms: {heuristic_metrics['false_positives']}")

    improvement = ((ml_pr_auc - heuristic_pr_auc) / heuristic_pr_auc) * 100 if heuristic_pr_auc > 0 else 0.0

    print(f"\n--- Machine Learning Model ({args.model.upper()}) ---")
    print(f"PR-AUC:   {ml_pr_auc:.4f}  <-- {improvement:.1f}% Improvement over baseline")

    print(f"\nMetrics at Default Config Threshold ({config.ALERT_THRESHOLD}):")
    print(f"- F1-Score:  {ml_default_metrics['f1_score']:.4f}")
    print(f"- F2-Score:  {ml_default_metrics['f2_score']:.4f}")

    print(f"\nMetrics at Optimal SRE Threshold ({optimal_thresh:.4f}):")
    print(f"- Precision: {ml_optimal_metrics['precision']:.4f}")
    print(f"- Recall:    {ml_optimal_metrics['recall']:.4f}")
    print(f"- F1-Score:  {ml_optimal_metrics['f1_score']:.4f}")
    print(f"- F2-Score:  {ml_optimal_metrics['f2_score']:.4f}")
    print(f"Confusion Matrix:")
    print(f"  Caught: {ml_optimal_metrics['true_positives']} | Missed: {ml_optimal_metrics['false_negatives']} | False Alarms: {ml_optimal_metrics['false_positives']}")


    # In a production system, raw probability crossing a threshold causes pager storms.
    # We apply a Stateful Cooldown heuristic to suppress duplicate alerts for ongoing incidents.
    print("\n[4] Real-World Adaptation: Alert Fatigue & Statefulness...")
    raw_alerts = (ml_prob >= optimal_thresh).astype(int)
    total_raw_pages = np.sum(raw_alerts)

    cooldown_period = 30
    stateful_alerts = simulate_stateful_alerts(ml_prob, optimal_thresh, cooldown_steps=cooldown_period)
    total_stateful_pages = np.sum(stateful_alerts)

    reduction = ((total_raw_pages - total_stateful_pages) / total_raw_pages) * 100 if total_raw_pages > 0 else 0

    print(f"\nResults using Optimal Threshold ({optimal_thresh:.4f}):")
    print(f"  Without Cooldown (Raw ML): {total_raw_pages} discrete pages sent.")
    print(f"  With {cooldown_period}-Min Cooldown:     {total_stateful_pages} pages sent.")
    print(f"  Reduction in Alert Spam:   {reduction:.1f}%")


    print("\n[5] Generating Visualizations...")
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    precision_devops, recall_devops, _ = precision_recall_curve(y_test, heuristic_prob)
    ax.plot(recall_devops, precision_devops, color='#ff7f0e', linestyle='--', label="DevOps Threshold")

    precision_ml, recall_ml, _ = precision_recall_curve(y_test, ml_prob)
    ax.plot(recall_ml, precision_ml, color='#1f77b4', label=args.model.upper())

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    plt.title(f"Precision-Recall Curve: {args.model.upper()} vs Standard Alerting", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    pr_plot_path = f"results/pr_curve_{args.model}.png"
    plt.savefig(pr_plot_path, bbox_inches='tight')
    print(f"Saved plot to: {pr_plot_path}")
    print("\nEvaluation Complete.")


if __name__ == "__main__":
    main()
