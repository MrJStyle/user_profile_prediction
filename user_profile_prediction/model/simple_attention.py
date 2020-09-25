from typing import List

from numpy import array
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Bidirectional, Dense, LSTM, Permute, Multiply, Layer, Embedding


class SimpleAttention(Model):
    def __init__(self, sentence_len: int, embedding_size: int, class_num: int, embedding_matrix: array):
        super(SimpleAttention, self).__init__()
        self.sentence_len: int = sentence_len
        self.embedding_size: int = embedding_size
        self.class_num: int = class_num
        self.embedding_matrix: array = embedding_matrix

    def build(self, input_shape) -> None:
        self.embedding: Embedding = Embedding(input_dim=self.embedding_matrix.shape[0],
                                             output_dim=self.embedding_matrix.shape[1],
                                             weights=[self.embedding_matrix],
                                             input_length=self.sentence_len,
                                             trainable=False)
        self.permute_1: Permute = Permute((2, 1), name="permute_1")
        self.dence_1: Dense = Dense(self.sentence_len, activation="softmax", name="dense_1")
        self.attention: Permute = Permute((2, 1), name="attention")
        self.multiply: Multiply = Multiply(name="multiply")

        self.bidirectionnal_1: Bidirectional = Bidirectional(
            LSTM(units=10, dropout=0.2, return_sequences=True),
            name="bidirectional_1"
        )

        self.bidirectionnal_2: Bidirectional = Bidirectional(
            LSTM(units=10, dropout=0.2),
            name="bidirectional_2"
        )

        self.dence_2: Dense = Dense(10, activation="relu", name="dense_2")
        self.dence_3: Dense = Dense(self.class_num, activation="softmax", name="dense_3")
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs, training=None, mask=None) -> Tensor:
        inputs = self.embedding(inputs)
        res: Tensor = self.permute_1(inputs)
        res: Tensor = self.dence_1(res)
        res: Tensor = self.attention(res)
        res: Tensor = self.multiply([inputs, res])

        res = self.bidirectionnal_1(res)
        res = self.bidirectionnal_2(res)

        res = self.dence_2(res)
        res = self.dence_3(res)
        return res

    def summary(self, line_length=None, positions=None, print_fn=None) -> None:
        inputs: Input = Input(shape=(self.sentence_len,), name="Input")
        outputs = self.call(inputs)

        model = Model(inputs=inputs, outputs=outputs, name="simple_attention")
        return model.summary()


if __name__ == "__main__":
    import datetime
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.metrics import Mean, CategoricalAccuracy
    from tensorflow.keras.callbacks import TensorBoard

    from user_profile_prediction.etl.embedding import Embedding as E
    from user_profile_prediction.etl.preprocess_train_data import PreprocessTrainingData
    from user_profile_prediction.training.deep_learning_model_train_step import DeepLearningModelTraining

    p: PreprocessTrainingData = PreprocessTrainingData(
        "/Volumes/Samsung_T5/Files/Document/china_hadoop/GroupProject/project_data/data/train.csv",
        embedding_size=10,
        sentence_len=10
    )
    p.split_sentence()

    e = E(10, 1)
    m = e.train_word2vec_model(p.sentences_with_split_words)

    x_train, x_val, y_train, y_val = p.split_data(p.age_data_iter(e))

    attention = SimpleAttention(10, 10, 6, e.embedding_matrix)

    optimizer: SGD = SGD(learning_rate=1e-3)
    losses: CategoricalCrossentropy = CategoricalCrossentropy()
    metric = CategoricalAccuracy()

    step = DeepLearningModelTraining(attention, e, optimizer, losses, metric)

    step.build((None, 10, ))
    step.compile()
    step.fit(x_train, y_train, x_val, y_val, epochs=30, batch=500)
