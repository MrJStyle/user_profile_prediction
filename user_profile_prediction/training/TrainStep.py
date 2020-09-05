import tensorflow as tf

from tensorflow import Tensor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from typing import Type

from user_profile_prediction.training import Optimizer, Losses, Metrics
from user_profile_prediction.model import TrainingModel
from user_profile_prediction.etl import EmbeddingModel


class ModelTraining:
    _training_model: TrainingModel
    _embedding_model: EmbeddingModel

    def __init__(
            self,
            training_model: TrainingModel,
            embedding_model: EmbeddingModel,

    ):
        self.training_model = training_model
        self.embedding_model = embedding_model

    @property
    def training_model(self):
        return getattr(self, "_training_model", None)

    @training_model.setter
    def training_model(self, model):
        if not isinstance(model, TrainingModel):
            raise TypeError("error training model type")

        self._training_model = model

    @property
    def embedding_model(self):
        return getattr(self, "_embedding_model", None)

    @embedding_model.setter
    def embedding_model(self, model):
        if not isinstance(model, EmbeddingModel):
            raise TypeError("error embedding model type")

        self._embedding_model = model

    def single_train_step(self, features: Tensor, labels: Tensor):
        with tf.GradientTape() as tape:
            prediction: Tensor = self.training_model(features)


if __name__ == "__main__":
    pass