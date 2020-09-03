import os

from gensim.models import Word2Vec
from numpy import array
from pandas import DataFrame
from typing import List, Dict, Iterable, Tuple, Generator
from tensorflow import Tensor

from user_profile_prediction.etl.embedding import EmbeddingModel

current_file_path: str = os.path.abspath(__file__)
dir_path: str = os.path.dirname(current_file_path)


class BasePreprocess(object):
    data: DataFrame
    preprocess_data: DataFrame = DataFrame()

    sentences_with_split_words: List = list()

    def __init__(self, file_path: str):
        self.data = self.load_from_csv(file_path)

    @classmethod
    def load_from_csv(cls, file_path: str) -> DataFrame: ...

    @staticmethod
    def filter_stop_words(words: Iterable) -> List: ...

    def age_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]: ...

    def gender_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]: ...

    def education_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]: ...

    @staticmethod
    def trans_data_to_tensor(data_iter: Generator) -> Generator[Tuple[Tensor, Tensor], None, None]: ...


class BaseEmbedding:
    MIN_COUNT: int
    EMBEDDING_SIZE: int
    EMBEDDING_MODEL_SAVED_PATH: str = os.path.join(dir_path, "embedding_model.txt")

    _embedding_model: Word2Vec

    def train_word2vec_model(self, sentences_with_spilt_words: Iterable[Iterable[str]]) -> Word2Vec: ...
    def load_embedding_model(self) -> Word2Vec: ...
    def words_to_vec(self, words: Iterable[str], sentence_len: int) -> array: ...
