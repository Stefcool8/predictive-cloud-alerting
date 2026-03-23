"""
Central registry for instantiating models.
Allows unified training and evaluation scripts to swap models dynamically.
"""

from src.models.hybrid_model import HybridAlertingModel
from src.models.random_forest import RandomForestModel


def get_model(model_name: str):
    """
    Returns an uninitialized instance of the requested model.
    """
    registry = {
        "rf": RandomForestModel,
        "hybrid": HybridAlertingModel,
        # Future models go here (e.g., "xgboost": XGBoostAlertingModel)
    }

    if model_name not in registry:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(registry.keys())}")

    return registry[model_name]()
