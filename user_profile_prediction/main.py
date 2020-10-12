import datetime
from typing import Optional

import click
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from user_profile_prediction.etl.preprocess_train_data import PreprocessTrainingData
from user_profile_prediction.model import TextCNN, FastText
from user_profile_prediction.training.deep_learning_model_train_step import DeepLearningModelTraining

TRAINING_FILE_PATH: str
TEST_FILE_PATH: str

MODEL: Optional[str] = None

EMBEDDING_SIZE: int
SENTENCE_LEN: int
VOCABULARY_SIZE: int

MIN_COUNT: int
CLASS_NUM: int

LEARNING_RATE: float
EPOCHS: int
BATCH_SIZE: int


class Model(object):
    def __new__(cls, *args, **kwargs):
        if MODEL is None:
            raise ValueError("no select any model")

        if MODEL == "TextCNN":
            return TextCNN(SENTENCE_LEN, EMBEDDING_SIZE, CLASS_NUM, VOCABULARY_SIZE)

        if MODEL == "FastText":
            return FastText(SENTENCE_LEN, EMBEDDING_SIZE, CLASS_NUM, VOCABULARY_SIZE)

        raise ValueError("error model name")


@click.group()
def cli():
    pass


@cli.command(name="TextCNN")
@click.option("--training_file_path", type=str)
@click.option("--embedding_size", type=int)
@click.option("--sentence_len", type=int)
@click.option("--conv_filter", type=int)
@click.option("--global_max_pool", type=bool)
@click.option("--pool_size", type=int)
@click.option("--drop_rate", type=float)
@click.option("--dense_size", type=int)
@click.option("--l1_regularization", type=float)
@click.option("--l2_regularization", type=float)
@click.option("--vocabulary_size", type=int)
@click.option("--min_count", type=int)
@click.option("--class_num", type=int)
@click.option("--label_name", type=str)
@click.option("--learning_rate", type=float)
@click.option("--epochs", type=int)
@click.option("--batch_size", type=int)
@click.option("--checkpoint_saved_path", type=str)
def training_model(
        training_file_path: str,
        embedding_size: int,
        sentence_len: int,
        conv_filter: int,
        global_max_pool: bool,
        pool_size: int,
        drop_rate: float,
        dense_size: int,
        l1_regularization: float,
        l2_regularization: float,
        vocabulary_size: int,
        min_count: int,
        text_cnn,
        class_num: int,
        label_name: int,
        learning_rate: int,
        epochs: int,
        batch_size: int,
        checkpoint_saved_path: str
):
    global TRAINING_FILE_PATH, EMBEDDING_SIZE, SENTENCE_LEN, MIN_COUNT, CLASS_NUM, LEARNING_RATE,\
        EPOCHS, BATCH_SIZE, MODEL, VOCABULARY_SIZE

    TRAINING_FILE_PATH = training_file_path
    EMBEDDING_SIZE = embedding_size
    SENTENCE_LEN = sentence_len
    VOCABULARY_SIZE = vocabulary_size

    MIN_COUNT = min_count
    CLASS_NUM = class_num

    MODEL = text_cnn

    LEARNING_RATE = learning_rate
    EPOCHS = epochs
    BATCH_SIZE = batch_size

    preprocess: PreprocessTrainingData = PreprocessTrainingData(
        csv_file_path=TRAINING_FILE_PATH,
        embedding_size=EMBEDDING_SIZE,
        sentence_len=SENTENCE_LEN,
        vocabulary_size=VOCABULARY_SIZE
    )

    preprocess.split_sentence()

    x_train, x_val, y_train, y_val = preprocess.split_data(preprocess.age_data_iter())

    text_cnn = TextCNN(
        SENTENCE_LEN, EMBEDDING_SIZE, CLASS_NUM, VOCABULARY_SIZE, conv_filter, global_max_pool, pool_size,
        drop_rate, dense_size, l1_regularization, l2_regularization
    )

    optimizer: Adam = Adam(learning_rate=LEARNING_RATE)
    losses: CategoricalCrossentropy = CategoricalCrossentropy()
    metric = CategoricalAccuracy()

    training = DeepLearningModelTraining(text_cnn, optimizer, losses, metric)

    training.build((None, SENTENCE_LEN, ))
    training.compile()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    training.fit(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        callbacks=[tensorboard_callback]
    )

    ckpt = tf.train.Checkpoint(text_cnn=training.training_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_saved_path, max_to_keep=5)
    path = ckpt_manager.save()
    print("checkpoint of text_cnn has been saved in {}".format(path))


if __name__ == "__main__":
    cli()
