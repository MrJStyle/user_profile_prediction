from typing import List

from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Concatenate, Dense, Flatten, Dropout


class TextRNN(Model):
    def __init__(self, sentence_len: int, embedding_size, class_num: int):
        super(TextRNN, self).__init__()
        self.sentence_len: int = sentence_len
        self.embedding_size: int = embedding_size
        self.class_num: int = class_num

    def build(self, input_shape) -> None:
        self.birectional: Bidirectional = Bidirectional(LSTM(128), name="bidirectional")
        # self.dense_1: Dense = Dense(100, name="dense_1")
        self.dropout_1: Dropout = Dropout(0.5, name="dropout_1")
        self.dense_2: Dense = Dense(6, name="dense_2")
        super(TextRNN, self).build(input_shape)

    def call(self, inputs, training=None, mask=None) -> Tensor:
        res = self.birectional(inputs)
        # res = self.dense_1(res)
        res = self.dropout_1(res)
        res = self.dense_2(res)

        return res

    def summary(self, line_length=None, positions=None, print_fn=None):
        input: Input = Input(shape=(self.sentence_len, self.embedding_size), name="Input")
        output: Tensor = self.call(input)
        model: Model = Model(inputs=input, outputs=output, name="TextRNN")
        model.summary()


if __name__ == "__main__":
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.metrics import Mean, CategoricalAccuracy

    from user_profile_prediction.etl.embedding import Embedding
    from user_profile_prediction.etl.preprocess_train_data import PreprocessTrainingData
    from user_profile_prediction.training.deep_learning_model_train_step import DeepLearningModelTraining

    p: PreprocessTrainingData = PreprocessTrainingData(
        "/Volumes/Samsung_T5/Files/Document/china_hadoop/GroupProject/project_data/data/train.csv", embedding_size=500, sentence_len=400)
    p.split_sentence()

    e = Embedding(500, 10)
    m = e.train_word2vec_model(p.sentences_with_split_words)
    # m = e.load_embedding_model()

    x_train, x_val, y_train, y_val = p.split_data(p.age_data_iter(e))

    text_rnn = TextRNN(400, 500, 6)

    optimizer: Adam = Adam(learning_rate=1e-4)
    losses: CategoricalCrossentropy = CategoricalCrossentropy()
    metric = CategoricalAccuracy()

    step = DeepLearningModelTraining(text_rnn, e, optimizer, losses, metric)

    step.build((None, 400, 500))
    step.compile()

    step.fit(x_train, y_train, x_val, y_val, 500, 50)