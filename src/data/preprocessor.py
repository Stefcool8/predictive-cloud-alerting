"""
Logic for converting continuous time-series data into discrete sliding windows
for supervised machine learning.
"""

import numpy as np
import pandas as pd

from src import config


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = config.WINDOW_SIZE,
    horizon: int = config.HORIZON,
    predict_new_only: bool = config.PREDICT_NEW_ONLY
):
    """
    Transforms a time-series dataframe into X (features) and Y (labels) arrays.
    """
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    feature_cols = [col for col in df.columns if col != 'is_incident']
    features_array = df[feature_cols].values
    labels_array = df['is_incident'].values

    x, y = [], []

    total_steps = len(df)
    max_valid_index = total_steps - horizon

    print(f"Creating sliding windows (W={window_size}, H={horizon})...")

    for i in range(window_size, max_valid_index):
        # Determine the state of the system at the exact moment of prediction
        # (The very last minute of the historical window W)
        current_state = labels_array[i - 1]

        # If the system is already broken, skip this sample.
        # The model should only predict the start of an outage.
        if predict_new_only and current_state == 1:
            continue

        window_x = features_array[i - window_size: i]
        window_y = labels_array[i: i + horizon]

        label = 1 if np.any(window_y == 1) else 0

        x.append(window_x)
        y.append(label)

    return np.array(x), np.array(y)


if __name__ == "__main__":
    from src.data.generator import generate_synthetic_data
    print("Generating small dataset for testing...")
    df_sample = generate_synthetic_data(num_timesteps=5000)
    X, Y = create_sliding_windows(df_sample)

    print("\n--- Preprocessing Results ---")
    print(f"Input Data Shape: {df_sample.shape}")
    print(f"X (Features) Shape: {X.shape}")
    print(f"Y (Labels) Shape: {Y.shape}")
    print(f"Number of Positive Samples (Alerts to fire): {np.sum(Y)}")
    print(f"Class Imbalance: {np.sum(Y) / len(Y) * 100:.2f}% positive")
