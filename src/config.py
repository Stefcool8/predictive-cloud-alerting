"""
Configuration settings for the Predictive Cloud Alerting project.
"""

# --- Time Series & Sliding Window Parameters ---
WINDOW_SIZE = 60
HORIZON = 15

# --- Synthetic Data Generation Parameters ---
TOTAL_TIMESTEPS = 500000
NUM_FEATURES = 5
ANOMALY_PROB = 0.001
ANOMALY_DURATION = 10

# --- Model & Training Parameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
PREDICT_NEW_ONLY = True
ALERT_THRESHOLD = 0.7

# --- File Paths ---
RAW_DATA_PATH = "data/raw/synthetic_metrics.csv"
PROCESSED_X_PATH = "data/processed/X_features.npy"
PROCESSED_Y_PATH = "data/processed/Y_labels.npy"

# Centralized model paths for easy expansion
MODEL_PATHS = {
    "rf": "models_saved/rf_model.pkl",
    "hybrid": "models_saved/hybrid_model.pt"
}
