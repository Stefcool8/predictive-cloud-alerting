"""
Execution script to generate synthetic cloud metrics, apply the sliding window
preprocessing, and save the resulting datasets to disk for training.

Run this from the root of the project:
$ python scripts/generate_data.py
"""

import os

import numpy as np

from src import config
from src.data.generator import generate_synthetic_data
from src.data.preprocessor import create_sliding_windows


def main():
    print("=== Phase 1: Generating Raw Synthetic Time-Series ===")
    df_raw = generate_synthetic_data(
        num_timesteps=config.TOTAL_TIMESTEPS,
        num_features=config.NUM_FEATURES,
        anomaly_prob=config.ANOMALY_PROB,
        anomaly_duration=config.ANOMALY_DURATION
    )

    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    df_raw.to_csv(config.RAW_DATA_PATH, index=False)

    print(f"Raw data saved to: {config.RAW_DATA_PATH}")
    print(f"Total rows: {len(df_raw)}, Total incidents injected: {df_raw['is_incident'].sum()}")

    print("\n=== Phase 2: Applying Sliding Window (W) and Horizon (H) ===")
    # Always outputs 3D tensors (Samples, Timesteps, Features) for universal model compatibility
    x, y = create_sliding_windows(
        df_raw,
        window_size=config.WINDOW_SIZE,
        horizon=config.HORIZON,
        predict_new_only=config.PREDICT_NEW_ONLY
    )

    os.makedirs(os.path.dirname(config.PROCESSED_X_PATH), exist_ok=True)
    np.save(config.PROCESSED_X_PATH, x)
    np.save(config.PROCESSED_Y_PATH, y)

    print(f"Processed features (X) saved to: {config.PROCESSED_X_PATH} | Shape: {x.shape}")
    print(f"Processed labels (Y) saved to: {config.PROCESSED_Y_PATH}   | Shape: {y.shape}")

    positive_samples = np.sum(y)
    total_samples = len(y)
    imbalance_ratio = (positive_samples / total_samples) * 100

    print(f"\nDataset Class Balance:")
    print(f"- Negative Samples (Normal): {total_samples - positive_samples}")
    print(f"- Positive Samples (Alerts): {positive_samples}")
    print(f"- Imbalance: {imbalance_ratio:.2f}% positive samples")
    print("\nData generation pipeline complete! Ready for model training.")


if __name__ == "__main__":
    main()
