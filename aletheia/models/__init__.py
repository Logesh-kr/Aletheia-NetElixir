"""
aletheia.models
===============
Model training, inference, and persistence for the Aletheia ML pipeline.

Public API
----------
    from aletheia.models.base import BaseModel
    from aletheia.models.lightgbm_model import LightGBMModel
    from aletheia.models.trainer import ModelTrainer, TrainingResult
    from aletheia.models.predictor import ModelPredictor
"""

from .base import BaseModel
from .lightgbm_model import LightGBMModel
from .trainer import ModelTrainer, TrainingResult
from .predictor import ModelPredictor

__all__ = [
    "BaseModel",
    "LightGBMModel",
    "ModelTrainer",
    "TrainingResult",
    "ModelPredictor",
]
