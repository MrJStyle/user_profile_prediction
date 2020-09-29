from array import array
from typing import List

from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, GlobalAveragePooling1D


class FastText(Model):
    def __init__(self, sentence_len: int, embedding_size: int, class_num: int, embedding_matrix: array):
        super(FastText, self).__init__()
        self.sentence_len: int = sentence_len
        self.embedding_size: int = embedding_size
        self.class_num: int = class_num
        self.embedding_matrix: array = embedding_matrix

    def build(self, input_shape) -> None:
        self.embedding: Embedding = Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            input_length=self.sentence_len,
            trainable=False
        )
        self.dropout_1: Dropout = Dropout(0.5, name="dropout_1")
        self.global_average_pool: GlobalAveragePooling1D = GlobalAveragePooling1D(name="average_pool_1")
        self.dropout_2: Dropout = Dropout(0.5, name="dropout_2")
        self.dense: Dense = Dense(self.class_num, activation="softmax", kernel_regularizer=regularizers.l2(0.001))

        super(FastText, self).build(input_shape)

    def call(self, inputs: Dataset, training=None, mask=None) -> Tensor:
        res = self.embedding(inputs)
        res = self.dropout_1(res)
        res = self.global_average_pool(res)
        res = self.dropout_2(res)
        res = self.dense(res)

        return res

    def summary(self, line_length=None, positions=None, print_fn=None) -> None:
        input: Input = Input(shape=(self.sentence_len, ), name="Input")
        output = self.call(input)
        model = Model(inputs=input, outputs=output, name="FastText")
        model.summary()

if __name__ == "__main__":
    import datetime
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.metrics import Mean, CategoricalAccuracy
    from tensorflow.keras.callbacks import TensorBoard

    from user_profile_prediction.etl.embedding import Embedding as E
    from user_profile_prediction.etl.preprocess_train_data import PreprocessTrainingData
    from user_profile_prediction.training.deep_learning_model_train_step import DeepLearningModelTraining

    EMBEDDING_SIZE = 150
    SENTENCE_LEN = 400
    MIN_COUNT = 1
    CLASS_NUM = 6

    LEARNING_RATE = 1e-3
    EPOCHS = 50
    BATCH_SIZE = 100

    p: PreprocessTrainingData = PreprocessTrainingData(
        "/home/mrj/Sundry/user-profile/train.csv",
        embedding_size=EMBEDDING_SIZE,
        sentence_len=SENTENCE_LEN)
    p.split_sentence()

    e = E(EMBEDDING_SIZE, MIN_COUNT)
    m = e.train_doc2vec_model(p.sentences_with_split_words, p.age_label)

    # m = e.load_embedding_model()

    x_train, x_val, y_train, y_val = p.split_data(p.age_data_iter(e))

    text_cnn = FastText(SENTENCE_LEN, EMBEDDING_SIZE, CLASS_NUM, e.embedding_matrix)

    optimizer: Adam = Adam(learning_rate=LEARNING_RATE)
    losses: CategoricalCrossentropy = CategoricalCrossentropy()
    metric = CategoricalAccuracy()

    step = DeepLearningModelTraining(text_cnn, e, optimizer, losses, metric)

    step.build((None, SENTENCE_LEN, ))
    step.compile()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    step.fit(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        callbacks=[tensorboard_callback]
    )