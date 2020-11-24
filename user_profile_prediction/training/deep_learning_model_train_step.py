from typing import List, Union, Tuple

import tensorflow as tf
from tensorflow import Tensor

from user_profile_prediction.etl import EmbeddingModel
from user_profile_prediction.model import TrainingModel
from user_profile_prediction.training import Optimizer, Losses, Metrics


class DeepLearningModelTraining:
    _training_model: TrainingModel
    _embedding_model: EmbeddingModel
    _optimizer: Optimizer
    _losses: Losses
    _metric: Union[Metrics, List[Metrics]]

    def __init__(
            self,
            training_model: TrainingModel,
            # embedding_model: EmbeddingModel,   # embedding model that has been trained
            optimizer: Optimizer = None,
            losses: Losses = None,
            metrics: Union[Metrics, List[Metrics]] = None
    ):
        self.training_model = training_model
        # self.embedding_model = embedding_model
        self._optimizer = optimizer
        self._losses = losses
        self._metric = metrics

    @property
    def training_model(self) -> TrainingModel:
        return getattr(self, "_training_model", None)

    @training_model.setter
    def training_model(self, model):
        # if not isinstance(model, TrainingModel):
        #     raise TypeError("error training model type")

        self._training_model = model

    @property
    def embedding_model(self) -> EmbeddingModel:
        return getattr(self, "_embedding_model", None)

    @embedding_model.setter
    def embedding_model(self, model):
        # if not isinstance(model, EmbeddingModel):
        #     raise TypeError("error embedding model type")

        self._embedding_model = model

    def build(self, input_shape: Tuple):
        self._training_model.build(input_shape)

    def compile(self, *args, **kwargs) -> None:
        self._training_model.compile(
            optimizer=self._optimizer,
            loss=self._losses,
            metrics=self._metric,
            *args,
            **kwargs
        )
        self._training_model.summary()

    def fit(
            self,
            x_train: Tensor,
            y_train,
            x_val: Tensor,
            y_val: Tensor,
            epochs: int = 20,
            batch: int = 100,
            *args,
            **kwargs
    ):
        self._training_model.fit(
            x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch, *args, **kwargs
        )

    def single_train_workflow(self, features: Tensor, labels: Tensor):
        with tf.GradientTape() as tape:
            prediction: Tensor = self.training_model(features)
            loss = self._losses.__call__(labels, prediction)

        gradient: List[Tensor] = tape.gradient(loss, self._training_model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradient, self._training_model.trainable_variables))

        self._metric.update_state(labels, prediction)

    # def main_train_workflow(self, x_train: ):

