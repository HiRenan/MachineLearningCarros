"""
Utility Functions for Model Training and Evaluation
"""

import numpy as np
import pandas as pd
import joblib
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape)
    }

def inverse_transform_predictions(y_pred_transformed: np.ndarray, transform_method: str = "log1p") -> np.ndarray:
    if transform_method == "log1p":
        return np.expm1(y_pred_transformed)
    return y_pred_transformed

def save_model_package(model: Any, model_name: str, metrics: Dict, feature_names: list, output_dir: str = "../models") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_package = {
        "model": model,
        "model_name": model_name,
        "feature_names": feature_names,
        "metadata": {
            "training_date": datetime.now().isoformat(),
            "metrics": metrics
        }
    }
    
    filepath = output_path / f"{model_name}_package.pkl"
    joblib.dump(model_package, filepath)
    return str(filepath)

def load_model_package(filepath: str) -> Dict[str, Any]:
    return joblib.load(filepath)
