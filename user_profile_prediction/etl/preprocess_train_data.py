import os
import jieba
import pandas as pd
import numpy as np

from typing import List, Iterable, Tuple, Generator
from pandas import DataFrame
from numpy import array
from gensim.models import Word2Vec
from tensorflow import Tensor

from user_profile_prediction.etl import BasePreprocess
from user_profile_prediction.data.stopwords import StopwordsDataset

from tqdm import tqdm


stop_words: StopwordsDataset = StopwordsDataset()


class PreprocessTrainingData(BasePreprocess):
    embedding_model: Word2Vec

    MIN_COUNT: int = 5          # embedding
    SIZE: int = 100
    SENTENCE_LEN: int = 3
    EMBEDDING_MODEL_SAVED_PATH: str = os.path.join(BasePreprocess.dir_path, "embedding_model.txt")

    age_label: List[int] = list()
    gender_label: List[int] = list()
    education_label: List[int] = list()

    @classmethod
    def load_from_csv(cls, file_path: str) -> DataFrame:
        df: DataFrame = pd.read_csv(file_path, sep="###__###", header=None)
        df.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']

        return df

    def __init__(self, file_path: str, load_model: bool = False):
        super(PreprocessTrainingData, self).__init__(file_path)
        self.load_model = load_model
        self.preprocess_data.columns = list()

    def train_word2vec_model(self) -> Word2Vec:
        if self.load_model:
            self.load_embedding_model()
            return self.embedding_model

        for index, query in tqdm(self.data.iterrows()):
            if index == 1000:
                break

            for sentence in query["Query_List"].split("\t"):
                self.age_label.append(query["Age"])
                self.gender_label.append(query["Gender"])
                self.education_label.append(query["Education"])

                cut_words: List = jieba.lcut(sentence)
                self.split_word_sentences.append(self.filter_stop_words(cut_words))

        self.embedding_model: Word2Vec = Word2Vec(
            self.split_word_sentences, min_count=self.MIN_COUNT, size=self.SIZE
        )

        if os.path.exists(self.EMBEDDING_MODEL_SAVED_PATH):
            os.remove(self.EMBEDDING_MODEL_SAVED_PATH)

        self.embedding_model.save(self.EMBEDDING_MODEL_SAVED_PATH)

        return self.embedding_model

    @staticmethod
    def filter_stop_words(words: Iterable) -> List:
        return [w for w in words if w not in stop_words and w != " "]

    def load_embedding_model(self) -> None:
        self.embedding_model = Word2Vec.load(self.EMBEDDING_MODEL_SAVED_PATH)

    def words_to_vec(self, words: Iterable[str]) -> array:
        sentence_array: array = np.zeros((self.SENTENCE_LEN, self.SIZE))

        for i, w in enumerate(words):
            if i == self.SENTENCE_LEN:
                return sentence_array

            if w in self.embedding_model.wv:
                sentence_array[i] = self.embedding_model.wv[w]

        return sentence_array

    def age_data_iter(self) -> Generator[Tuple[array, int]]:
        for i, s in enumerate(self.split_word_sentences):
            yield self.words_to_vec(s), self.age_label[i]

    def gender_data_iter(self) -> Generator[Tuple[array, int]]:
        for i, s in enumerate(self.split_word_sentences):
            yield self.words_to_vec(s), self.gender_label[i]

    def education_data_iter(self) -> Generator[Tuple[array, int]]:
        for i, s in enumerate(self.split_word_sentences):
            yield self.words_to_vec(s), self.education_label[i]

    @staticmethod
    def trans_data_to_tensor(data_iter: Generator) -> Generator[Tuple[Tensor, Tensor]]:
        for x, y in data_iter:
            one_hot_y: Tensor = tf.one_hot(y, depth=tf.unique(y).y.shape[0])
            yield tf.constant(x), one_hot_y


if __name__ == "__main__":
    import tensorflow as tf

    p = PreprocessTrainingData("/Volumes/Samsung_T5/Files/Document/小象学院/GroupProject/project_data/data/train.csv")
    model = p.train_word2vec_model()

    ts = tf.data.Dataset.from_generator(p.age_data_iter, (tf.float16, tf.int8))

    print(list(ts.take(5).as_numpy_iterator()))
    print("finish")
