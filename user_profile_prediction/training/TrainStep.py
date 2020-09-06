import tensorflow as tf

from tensorflow import Tensor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from typing import List, Union

from user_profile_prediction.training import Optimizer, Losses, Metrics
from user_profile_prediction.model import TrainingModel
from user_profile_prediction.etl import EmbeddingModel


class ModelTraining:
    _training_model: TrainingModel
    _embedding_model: EmbeddingModel
    _optimizer: Optimizer
    _losses: Losses
    _metric: Union[Metrics, List[Metrics]]

    def __init__(
            self,
            training_model: TrainingModel,
            embedding_model: EmbeddingModel,   # embedding model that has been trained
            optimizer: Optimizer = None,
            losses: Losses = None,
            metrics: Union[Metrics, List[Metrics]] = None
    ):
        self.training_model = training_model
        self.embedding_model = embedding_model
        self._optimizer = optimizer
        self._losses = losses
        self._metric = metrics

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

    def compile(self, *args, **kwargs) -> None:
        self._training_model.compile(
            optimizer=self._optimizer,
            loss=self._losses,
            metrics=self._metric,
            *args,
            **kwargs
        )

    def fit(self):
        pass


    def single_train_workflow(self, features: Tensor, labels: Tensor):
        with tf.GradientTape() as tape:
            prediction: Tensor = self.training_model(features)
            loss = self._losses.__call__(labels, prediction)

        gradient: List[Tensor] = tape.gradient(loss, self._training_model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradient, self._training_model.trainable_variables))

        self._metric.update_state(labels, prediction)

    # def main_train_workflow(self, x_train: ):


if __name__ == "__main__":
    pass
