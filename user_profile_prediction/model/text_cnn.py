from typing import List

from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Conv1D, Concatenate, Dense, Flatten, Dropout, Embedding, \
    GlobalMaxPool1D, MaxPool1D


class TextCNN(Model):
    def __init__(
            self,
            sentence_len: int,
            embedding_size: int,
            class_num: int,
            vocabulary_size: int,
            conv_filter: int,
            global_max_pool: bool = True,
            pool_size: int = 2,
            drop_rate: float = 0.2,
            dense_size: int = 32,
            l1_regularization: float = 0.01,
            l2_regularization: float = 0.01
    ):
        """
        Parameters
        ----------
        sentence_len: the lens of tokenized sentence
        embedding_size: dim of embedding vector
        class_num: unique label num
        vocabulary_size: total words nums in vocabulary
        conv_filter: nums of filters in convolution layer
        global_max_pool: use global_max_pool_1D or max_pool_1D
        pool_size: pool size if choose to use max_pool_1D
        drop_rate: rate dropout layer
        dense_size: size of last second dense layer
        l1_regularization: penalty of l1
        l2_regularization: penalty of l2

        """
        super(TextCNN, self).__init__()
        self.sentence_len: int = sentence_len
        self.embedding_size: int = embedding_size
        self.class_num: int = class_num
        self.vocabulary_size: int = vocabulary_size
        self.conv_filter: int = conv_filter
        self.global_max_pool: bool = global_max_pool
        self.pool_size: int = pool_size
        self.drop_rate: float = drop_rate
        self.dense_size: int = dense_size
        self.l1_regularization: float = l1_regularization
        self.l2_regularization: float = l2_regularization

    def build(self, input_shape) -> None:
        self.embedding: Embedding = Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.embedding_size,
            input_length=self.sentence_len,
            trainable=True
        )
        self.conv_1: Conv1D = Conv1D(filters=self.conv_filter, kernel_size=3, activation="relu", name="conv_1")
        self.conv_2: Conv1D = Conv1D(filters=self.conv_filter, kernel_size=4, activation="relu", name="conv_2")
        self.conv_3: Conv1D = Conv1D(filters=self.conv_filter, kernel_size=5, activation="relu", name="conv_3")

        if self.global_max_pool:
            self.pool_1: MaxPool1D = MaxPool1D(pool_size=self.pool_size, strides=1, name="pool_1")
            self.pool_2: MaxPool1D = MaxPool1D(pool_size=self.pool_size, strides=1, name="pool_2")
            self.pool_3: MaxPool1D = MaxPool1D(pool_size=self.pool_size, strides=1, name="pool_3")
        else:
            self.pool_1: GlobalMaxPool1D = GlobalMaxPool1D(name="pool_1")
            self.pool_2: GlobalMaxPool1D = GlobalMaxPool1D(name="pool_2")
            self.pool_3: GlobalMaxPool1D = GlobalMaxPool1D(name="pool_3")

        self.concatenate: Concatenate = Concatenate(axis=1)
        self.flatten: Flatten = Flatten()

        self.dropout_1: Dropout = Dropout(self.drop_rate, name="dropout_1")
        self.dense1 = Dense(
            self.dense_size,
            activation="sigmoid",
            kernel_regularizer=regularizers.l1_l2(self.l1_regularization, self.l2_regularization)
        )
        self.dropout_2: Dropout = Dropout(self.drop_rate, name="dropout_2")
        self.dense: Dense = Dense(
            self.class_num,
            activation="softmax",
            kernel_regularizer=regularizers.l1_l2(self.l1_regularization, self.l2_regularization)
        )
        super(TextCNN, self).build(input_shape)

    def call(self, inputs: Dataset, training=None, mask=None) -> Tensor:
        inputs = self.embedding(inputs)
        convs: List[Tensor] = [
            self.conv_1(inputs),
            self.conv_2(inputs),
            self.conv_3(inputs)
        ]
        pools: List[Tensor] = [
            self.pool_1(convs[0]),
            self.pool_2(convs[1]),
            self.pool_3(convs[2])
        ]
        res: Tensor = self.concatenate(pools)
        res: Tensor = self.flatten(res)
        res: Tensor = self.dropout_1(res)
        res = self.dense1(res)
        res = self.dropout_2(res)
        res: Tensor = self.dense(res)
        return res

    def summary(self, line_length=None, positions=None, print_fn=None) -> None:
        input: Input = Input(shape=(self.sentence_len, ), name="Input")
        output = self.call(input)
        model = Model(inputs=input, outputs=output, name="TextCNN")
        model.summary()


if __name__ == "__main__":
    import datetime
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.metrics import CategoricalAccuracy
    from tensorflow.keras.callbacks import TensorBoard

    from user_profile_prediction.etl.embedding import Embedding as E
    from user_profile_prediction.etl.preprocess_train_data import PreprocessTrainingData
    from user_profile_prediction.training.deep_learning_model_train_step import DeepLearningModelTraining

    p: PreprocessTrainingData = PreprocessTrainingData(
        "/Volumes/Samsung_T5/Files/Document/china_hadoop/GroupProject/project_data/data/train.csv",
        embedding_size=150,
        sentence_len=400)
    p.split_sentence()

    e = E(150, 1)
    m = e.train_doc2vec_model(p.sentences_with_split_words, p.age_label)

    # m = e.load_embedding_model()

    x_train, x_val, y_train, y_val = p.split_data(p.age_data_iter(e))

    text_cnn = TextCNN(400, 150, 6, e.embedding_matrix)

    optimizer: Adam = Adam(learning_rate=5 * 1e-4)
    losses: CategoricalCrossentropy = CategoricalCrossentropy()
    metric = CategoricalAccuracy()

    step = DeepLearningModelTraining(text_cnn, e, optimizer, losses, metric)

    step.build((None, 400, ))
    step.compile()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    step.fit(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=50,
        batch=100,
        callbacks=[tensorboard_callback]
    )

    # import os
    # export_path: str = \
    #     "/Users/luominzhi/Code/Python/user_profile_prediction/user_profile_prediction/saved_model/text_cnn"
    # version = "1"
    #
    # step.training_model.save(os.path.join(export_path,  "{}".format(version)))
