import tensorflow as tf

from typing import List
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


class TextCNN(Model):
    def __init__(self, words_num: int, embedding_dim: int, class_num: int):
        super(TextCNN, self).__init__()
        self.input_layer: Input = Input((words_num, embedding_dim))
        self.conv_1: Conv1D = Conv1D(filters=128, kernel_size=1, activation="relu")
        self.pool_1: MaxPool1D = MaxPool1D()
        self.conv_2: Conv1D = Conv1D(filters=128, kernel_size=2, activation="relu")
        self.pool_2: MaxPool1D = MaxPool1D()
        self.conv_3: Conv1D = Conv1D(filters=128, kernel_size=3, activation="relu")
        self.pool_3: MaxPool1D = MaxPool1D()
        self.concatenate: Concatenate = Concatenate()
        self.dense: Dense = Dense(class_num ,activation="softmax")

    def call(self, inputs: Dataset, training=None, mask=None):
        x = self.input_layer(inputs)
        convs: List[Conv1D] = [self.conv_1(x), self.conv_2(x), self.conv_3(x)]
        pools: List[MaxPool1D] = [self.pool_1(convs[0]), self.pool_2(convs[1]), self.conv_3(convs[2])]
        x = self.concatenate(pools)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    from user_profile_prediction.etl.preprocess_train_data import PreprocessTrainingData
    p = PreprocessTrainingData("/Volumes/Samsung_T5/Files/Document/小象学院/GroupProject/project_data/data/train.csv")
    model = p.train_word2vec_model()

    ts = tf.data.Dataset.from_generator(p.age_data_iter, (tf.float16, tf.int8))

    text_cnn = TextCNN(PreprocessTrainingData.SENTENCE_LEN, PreprocessTrainingData.SIZE, 7)

    optimizer: Adam = Adam(learning_rate=1e-3)
    losses: CategoricalCrossentropy = CategoricalCrossentropy()

    text_cnn.compile(optimizer=optimizer, loss=losses)

    history = text_cnn.fit(ts, epochs=2, batch_size=64)

