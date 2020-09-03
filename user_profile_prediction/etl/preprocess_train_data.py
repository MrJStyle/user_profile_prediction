import os
import jieba
import pandas as pd
import numpy as np

from typing import List, Iterable, Tuple, Generator
from pandas import DataFrame
from numpy import array
from tensorflow import Tensor

from user_profile_prediction.etl import BasePreprocess
from user_profile_prediction.etl.embedding import EmbeddingModel
from user_profile_prediction.data.stopwords import StopwordsDataset

from tqdm import tqdm


stop_words: StopwordsDataset = StopwordsDataset()


class PreprocessTrainingData(BasePreprocess):
    age_label: List[int] = list()
    gender_label: List[int] = list()
    education_label: List[int] = list()

    @classmethod
    def load_from_csv(cls, file_path: str) -> DataFrame:
        df: DataFrame = pd.read_csv(file_path, sep="###__###", header=None)
        df.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']

        return df

    def __init__(self, file_path: str):
        super(PreprocessTrainingData, self).__init__(file_path)
        self.preprocess_data.columns = list()

    def split_sentence(self):
        for index, query in tqdm(self.data.iterrows()):
            if index == 1000:
                break

            for sentence in query["Query_List"].split("\t"):
                self.age_label.append(query["Age"])
                self.gender_label.append(query["Gender"])
                self.education_label.append(query["Education"])

                cut_words: List = jieba.lcut(sentence)
                self.sentences_with_split_words.append(self.filter_stop_words(cut_words))
        return self.sentences_with_split_words

    @staticmethod
    def filter_stop_words(words: Iterable) -> List:
        return [w for w in words if w not in stop_words and w != " "]

    def age_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]:
        for i, s in enumerate(self.sentences_with_split_words):
            yield model.words_to_vec(s, 3), self.age_label[i]

    def gender_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]:
        for i, s in enumerate(self.sentences_with_split_words):
            yield model.words_to_vec(s, 3), self.gender_label[i]

    def education_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]:
        for i, s in enumerate(self.sentences_with_split_words):
            yield model.words_to_vec(s, 3), self.education_label[i]

    @staticmethod
    def trans_data_to_tensor(data_iter: Generator) -> Generator[Tuple[Tensor, Tensor], None, None]:
        for x, y in data_iter:
            one_hot_y: Tensor = tf.one_hot(y, depth=tf.unique(y).y.shape[0])
            yield tf.constant(x), one_hot_y


if __name__ == "__main__":
    import tensorflow as tf
    from user_profile_prediction.etl.embedding import Embedding

    p = PreprocessTrainingData("/Volumes/Samsung_T5/Files/Document/小象学院/GroupProject/project_data/data/train.csv")
    p.split_sentence()

    e = Embedding(100, 5)
    m = e.load_embedding_model()

    next(p.age_data_iter(e))

    print("finish")
