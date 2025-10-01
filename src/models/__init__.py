"""
Models Module - Car Price Prediction

Este módulo contém classes e funções para treinamento e avaliação de modelos
de machine learning para predição de preços de carros.

Author: Machine Learning Pipeline
Date: 2024
"""

from .train import ModelTrainer
from .evaluate import ModelEvaluator
from .utils import (
    calculate_regression_metrics,
    inverse_transform_predictions,
    save_model_package,
    load_model_package
)

__all__ = [
    'ModelTrainer',
    'ModelEvaluator',
    'calculate_regression_metrics',
    'inverse_transform_predictions',
    'save_model_package',
    'load_model_package'
]

__version__ = '1.0.0'