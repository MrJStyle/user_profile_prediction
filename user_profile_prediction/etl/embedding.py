import os
from typing import Iterable, Dict

import numpy as np
from gensim.models import Word2Vec
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer

from user_profile_prediction.etl import BaseEmbedding
from user_profile_prediction.exceptions import NotInitError


class Embedding(BaseEmbedding):
    tokenizer: Tokenizer
    word_index: Dict
    embedding_matrix: array

    def __init__(self, embedding_size: int, min_count: int = 1, save_path: str = None):
        self.EMBEDDING_SIZE = embedding_size
        self.MIN_COUNT = min_count
        if save_path:
            self.EMBEDDING_MODEL_SAVED_PATH = save_path

    @property
    def embedding_model(self):
        if not self.embedding_model:
            raise NotInitError("please pre-train or load model first!")
        return self._embedding_model

    def train_word2vec_model(self, sentences_with_spilt_words: Iterable[Iterable[str]]) -> Word2Vec:
        self._embedding_model: Word2Vec = Word2Vec(
            sentences_with_spilt_words, min_count=self.MIN_COUNT, size=self.EMBEDDING_SIZE
        )

        if os.path.exists(self.EMBEDDING_MODEL_SAVED_PATH):
            os.remove(self.EMBEDDING_MODEL_SAVED_PATH)

        self._embedding_model.save(self.EMBEDDING_MODEL_SAVED_PATH)

        self.tokenizer: Tokenizer = Tokenizer(oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.tokenizer.fit_on_texts(sentences_with_spilt_words)
        self.word_index = self.tokenizer.word_index

        self.embedding_matrix = np.zeros_like((len(self.word_index) + 1, self.EMBEDDING_SIZE))

        for word, index in self.word_index.items():
            try:
                self.embedding_matrix[index] = self.embedding_matrix[word]
            except Exception:
                continue

        return self._embedding_model

    def load_embedding_model(self) -> Word2Vec:
        self._embedding_model = Word2Vec.load(self.EMBEDDING_MODEL_SAVED_PATH)
        return self._embedding_model

    def words_to_vec(self, words: Iterable[str], sentence_len: int) -> array:
        sentence_array: array = np.zeros((sentence_len, self.EMBEDDING_SIZE))

        for i, w in enumerate(words):
            if i == sentence_len:
                return sentence_array

            if w in self._embedding_model.wv:
                sentence_array[i] = self._embedding_model.wv[w]

        return sentence_array

    # def word_to_vec_with_tokenizer(self, words: Iterable[str], sentence_len: int, tokenizer: Tokenizer) -> array:
    #     word_index_map: dict = tokenizer.word_index
