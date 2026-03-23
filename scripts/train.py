"""
Unified training script.
Usage:
$ python -m scripts.train --model rf
$ python -m scripts.train --model cnn_lstm
"""

import argparse

import numpy as np

from src import config
from src.models.registry import get_model


def main():
    parser = argparse.ArgumentParser(description="Train a predictive alerting model.")
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'hybrid'],
                        help="The model architecture to train.")
    args = parser.parse_args()

    print("=== Phase 1: Loading Processed Data ===")
    x = np.load(config.PROCESSED_X_PATH)
    y = np.load(config.PROCESSED_Y_PATH)

    split_index = int(len(x) * (1 - config.TEST_SIZE))
    x_train, y_train = x[:split_index], y[:split_index]

    print(f"\n=== Phase 2: Training {args.model.upper()} ===")
    model = get_model(args.model)
    model.train(x_train, y_train)

    print("\n=== Phase 3: Saving Model ===")
    save_path = config.MODEL_PATHS[args.model]
    model.save_model(save_path)


if __name__ == "__main__":
    main()
