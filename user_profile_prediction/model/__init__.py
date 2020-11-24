from typing import NewType

from tensorflow.keras.models import Model

from user_profile_prediction.model.fasttext import FastText
from user_profile_prediction.model.text_cnn import TextCNN

TrainingModel = NewType("TrainingModel", Model)

__all__ = [
    "TrainingModel",
    "TextCNN",
    "FastText"
]

