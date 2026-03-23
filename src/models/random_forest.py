"""
Baseline machine learning model for predictive alerting.
Uses a Random Forest Classifier and securely handles feature scaling
to allow easy swapping with distance-based or deep learning models later.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src import config


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=10, random_state=config.RANDOM_STATE):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    @staticmethod
    def _flatten_3d(x: np.ndarray) -> np.ndarray:
        """Flattens (Samples, Timesteps, Features) to (Samples, Timesteps * Features)"""
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input, got {x.shape}")
        return x.reshape(x.shape[0], -1)

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Fits the scaler on the training data to prevent future data leakage,
        transforms the training data, and then trains the model.
        """
        print(f"Training baseline Random Forest model on {len(x_train)} samples...")

        x_flat = self._flatten_3d(x_train)
        # Fit the scaler only on the training data and transform it.
        x_train_scaled = self.scaler.fit_transform(x_flat)

        self.model.fit(x_train_scaled, y_train)
        self.is_trained = True
        print("Training complete.")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Scales the incoming data using the pre-fitted scaler.
        Returns the probability of an incident occurring in the future horizon.

        Returns:
            np.ndarray: 1D array of probabilities (0.0 to 1.0) for the positive class.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calling predict_proba.")

        x_flat = self._flatten_3d(x)
        # Only transform the data
        x_scaled = self.scaler.transform(x_flat)

        return self.model.predict_proba(x_scaled)[:, 1]

    def save_model(self, filepath: str):
        """
        Serializes both the trained model and the fitted scaler to disk.
        """
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save them together as a dictionary so they never get separated
        artifact = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(artifact, filepath)
        print(f"Model and Scaler successfully saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Loads the model and scaler artifacts from the disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved artifact found at {filepath}")

        artifact = joblib.load(filepath)
        self.model = artifact['model']
        self.scaler = artifact['scaler']
        self.is_trained = True
        print(f"Model and Scaler successfully loaded from {filepath}")


if __name__ == "__main__":
    model = RandomForestModel()
    print("Model initialized successfully. Ready for the training pipeline.")
