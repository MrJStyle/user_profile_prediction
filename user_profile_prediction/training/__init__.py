from typing import NewType
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.metrics import MeanMetricWrapper

Optimizer = NewType("Optimizer", OptimizerV2)
Losses = NewType("Losses", LossFunctionWrapper)
Metrics = NewType("Metrics", MeanMetricWrapper)

__all__ = [
    "Optimizer",
    "Losses",
    "Metrics"
]
