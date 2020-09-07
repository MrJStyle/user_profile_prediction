from typing import List

from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Concatenate, Dense, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


class TextCNN(Model):
    def __init__(self, class_num: int):
        super(TextCNN, self).__init__()
        self.class_num: int = class_num

    def build(self, input_shape) -> None:
        self.conv_1: Conv1D = Conv1D(filters=128, kernel_size=3, activation="relu", name="conv_1")
        self.pool_1: MaxPool1D = MaxPool1D(pool_size=2, strides=1, name="pool_1")
        self.conv_2: Conv1D = Conv1D(filters=128, kernel_size=4, activation="relu", name="conv_2")
        self.pool_2: MaxPool1D = MaxPool1D(pool_size=2, strides=1, name="pool_2")
        self.conv_3: Conv1D = Conv1D(filters=128, kernel_size=5, activation="relu", name="conv_3")
        self.pool_3: MaxPool1D = MaxPool1D(pool_size=2, strides=1, name="pool_3")
        self.concatenate: Concatenate = Concatenate(axis=1)
        self.flatten: Flatten = Flatten()

        self.dense1 = Dense(128, activation="sigmoid")
        self.dense: Dense = Dense(self.class_num, activation="softmax")
        super(TextCNN, self).build(input_shape)

    def call(self, inputs: Dataset, training=None, mask=None) -> Tensor:
        convs: List[Tensor] = [self.conv_1(inputs), self.conv_2(inputs), self.conv_3(inputs)]
        pools: List[Tensor] = [self.pool_1(convs[0]), self.pool_2(convs[1]), self.pool_3(convs[2])]
        res: Tensor = self.concatenate(pools)
        res: Tensor = self.flatten(res)
        res = self.dense1(res)
        res: Tensor = self.dense(res)
        return res

    def summary(self) -> None:
        input: Input = Input(shape=(100, 500), name="Input")
        output = self.call(input)
        model = Model(inputs=input, outputs=output, name="TextCNN")
        model.summary()


if __name__ == "__main__":
    from user_profile_prediction.etl.preprocess_train_data import PreprocessTrainingData
    p = PreprocessTrainingData("/home/mrj/Sundry/train.csv")
    model = p.train_word2vec_model()

    train_x, train_y = [], []
    age_data_iter = p.age_data_iter()

    for x, y in PreprocessTrainingData.trans_data_to_tensor(age_data_iter):
        train_x.append(x)
        train_y.append(y)

    text_cnn = TextCNN(7)

    optimizer: Adam = Adam(learning_rate=1e-3)
    losses: CategoricalCrossentropy = CategoricalCrossentropy()

    text_cnn.build((None, 3, 100))
    text_cnn.summary()

    text_cnn.compile(optimizer=optimizer, loss=losses)

    history = text_cnn.fit(train_x, train_y, epochs=50, batch_size=100)
