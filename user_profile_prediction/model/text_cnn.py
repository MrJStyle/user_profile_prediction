from typing import List

from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Concatenate, Dense, Flatten, Dropout


class TextCNN(Model):
    def __init__(self, sentence_len: int, embedding_size, class_num: int):
        super(TextCNN, self).__init__()
        self.sentence_len: int = sentence_len
        self.embedding_size: int = embedding_size
        self.class_num: int = class_num

    def build(self, input_shape) -> None:
        self.conv_1: Conv1D = Conv1D(filters=8, kernel_size=10, activation="relu", name="conv_1")
        self.pool_1: MaxPool1D = MaxPool1D(pool_size=4, strides=1, name="pool_1")
        self.conv_2: Conv1D = Conv1D(filters=8, kernel_size=20, activation="relu", name="conv_2")
        self.pool_2: MaxPool1D = MaxPool1D(pool_size=4, strides=1, name="pool_2")
        self.conv_3: Conv1D = Conv1D(filters=8, kernel_size=30, activation="relu", name="conv_3")
        self.pool_3: MaxPool1D = MaxPool1D(pool_size=4, strides=1, name="pool_3")
        self.concatenate: Concatenate = Concatenate(axis=1)
        self.flatten: Flatten = Flatten()

        self.dropout_1: Dropout = Dropout(0.5, name="dropout_1")
        self.dense1 = Dense(128, activation="sigmoid")
        self.dropout_2: Dropout = Dropout(0.5, name="dropout_2")
        self.dense: Dense = Dense(self.class_num, activation="softmax")
        super(TextCNN, self).build(input_shape)

    def call(self, inputs: Dataset, training=None, mask=None) -> Tensor:
        convs: List[Tensor] = [self.conv_1(inputs), self.conv_2(inputs), self.conv_3(inputs)]
        pools: List[Tensor] = [self.pool_1(convs[0]), self.pool_2(convs[1]), self.pool_3(convs[2])]
        res: Tensor = self.concatenate(pools)
        res: Tensor = self.flatten(res)
        res: Tensor = self.dropout_1(res)
        res = self.dense1(res)
        res = self.dropout_2(res)
        res: Tensor = self.dense(res)
        return res

    def summary(self, line_length=None, positions=None, print_fn=None) -> None:
        input: Input = Input(shape=(self.sentence_len, self.embedding_size), name="Input")
        output = self.call(input)
        model = Model(inputs=input, outputs=output, name="TextCNN")
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

    text_cnn = TextCNN(400, 500, 6)

    optimizer: Adam = Adam(learning_rate=1e-3)
    losses: CategoricalCrossentropy = CategoricalCrossentropy()
    metric = CategoricalAccuracy()

    step = DeepLearningModelTraining(text_cnn, e, optimizer, losses, metric)

    step.build((None, 400, 500))
    step.compile()

    step.fit(x_train, y_train, x_val, y_val, 4, 100)

    import os
    export_path: str = \
        "/Users/luominzhi/Code/Python/user_profile_prediction/user_profile_prediction/saved_model/text_cnn"
    version = "1"

    step.training_model.save(os.path.join(export_path,  "{}".format(version)))
