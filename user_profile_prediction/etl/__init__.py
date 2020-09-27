import os
import tensorflow as tf
from typing import List, Iterable, Tuple, Generator, NewType, Dict

from gensim.models import Word2Vec
from numpy import array
from pandas import DataFrame
from tensorflow import Tensor
from tensorflow.python.keras.preprocessing.text import Tokenizer

current_file_path: str = os.path.abspath(__file__)
dir_path: str = os.path.dirname(current_file_path)


class BaseEmbedding:
    MIN_COUNT: int
    EMBEDDING_SIZE: int
    EMBEDDING_MODEL_SAVED_PATH: str = os.path.join(dir_path, "embedding_model.txt")

    _embedding_model: Word2Vec

    def train_word2vec_model(self, sentences_with_spilt_words: Iterable[Iterable[str]]) -> Word2Vec: ...

    def load_embedding_model(self) -> Word2Vec: ...

    def words_to_vec(self, words: Iterable[str], sentence_len: int) -> array: ...


EmbeddingModel = NewType("EmbeddingModel", BaseEmbedding)


class BasePreprocess(object):
    data: DataFrame
    preprocess_data: DataFrame = DataFrame()

    sentences_with_split_words: List = list()
    sentences_with_split_words_sequence: List[List[int]]

    def __init__(self, csv_file_path: str):
        self.data: DataFrame = self.load_from_csv(csv_file_path)

    @classmethod
    def load_from_csv(cls, file_path: str) -> DataFrame: ...

    @staticmethod
    def filter_stop_words(words: Iterable) -> List: ...

    def age_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]: ...

    def gender_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]: ...

    def education_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]: ...

    @staticmethod
    def trans_data_to_tensor(data_iter: Generator) -> Generator[Tuple[Tensor, Tensor], None, None]:
        for x, y in data_iter:
            one_hot_y: Tensor = tf.one_hot(y, depth=tf.unique(y).y.shape[0])
            yield tf.constant(x), one_hot_y

    @staticmethod
    def concatenate_tensor(tensors: List[Tensor]) -> Tensor:
        return tf.concat(tensors, axis=0)

    @staticmethod
    def class_weights_to_penalty_weights(class_weights: Dict[int, int], sample_nums: int) -> Dict[int, float]:
        return {k: sample_nums / v for k, v in class_weights}


PreprocessModel = NewType("PreprocessModel", BasePreprocess)
