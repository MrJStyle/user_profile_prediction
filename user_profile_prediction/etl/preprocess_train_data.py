import jieba
import random
import pandas as pd
import numpy as np

from numpy import array
from pandas import DataFrame
from tensorflow import Tensor
from tqdm import tqdm
from typing import List, Iterable, Tuple, Generator, Collection

from user_profile_prediction.data.stopwords import StopwordsDataset
from user_profile_prediction.etl import BasePreprocess, EmbeddingModel

stop_words: StopwordsDataset = StopwordsDataset()


class PreprocessTrainingData(BasePreprocess):
    train_valid_test_weights: Collection

    age_label: List[int] = list()
    gender_label: List[int] = list()
    education_label: List[int] = list()

    @classmethod
    def load_from_csv(cls, file_path: str) -> DataFrame:
        df: DataFrame = pd.read_csv(file_path, sep="###__###", header=None)
        df.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']

        return df

    def __init__(self, file_path: str, train_valid_weights: Collection = (0.8, 0.2)):
        super(PreprocessTrainingData, self).__init__(file_path)
        self.preprocess_data.columns = list()
        self.train_valid_weights = train_valid_weights

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

    @property
    def train_valid_weights(self):
        return self._train_valid_weights

    @train_valid_weights.setter
    def train_valid_weights(self, weights: Collection):
        if weights.__len__() != 2:
            raise ValueError("set wrong dim weights")

        if sum(weights) != 1:
            raise ValueError("sum of weights not equal to 1")

        self._train_valid_weights = weights

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

    def split_data(self, data_iter: Generator) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        all_data: List[Tuple[Tensor, Tensor]]
        all_data = [d for d in self.trans_data_to_tensor(data_iter)]
        random.shuffle(all_data)

        split_pos: array = all_data.__len__() * np.array(self.train_valid_weights)
        split_pos = np.round(split_pos)

        train: List[Tuple[Tensor, Tensor]]
        valid: List[Tuple[Tensor, Tensor]]

        train, valid = all_data[:split_pos[0]], all_data[split_pos[0]: split_pos[1]]

        x_trains: List[Tensor]
        y_trains: List[Tensor]
        x_valids: List[Tensor]
        y_valids: List[Tensor]

        x_trains, y_trains = zip(*train)
        x_valids, y_valids = zip(*valid)

        x_train: Tensor
        y_train: Tensor
        x_valid: Tensor
        y_valid: Tensor

        x_train, y_train = self.concatenate_tensor(x_trains), self.concatenate_tensor(y_trains)
        x_valid, y_valid = self.concatenate_tensor(x_valids), self.concatenate_tensor(y_valids)

        return x_train, y_train, x_valid, y_valid


if __name__ == "__main__":
    import tensorflow as tf
    from user_profile_prediction.etl.embedding import Embedding

    p = PreprocessTrainingData("/Volumes/Samsung_T5/Files/Document/小象学院/GroupProject/project_data/data/train.csv")
    p.split_sentence()

    e = Embedding(100, 5)
    m = e.load_embedding_model()

    next(p.age_data_iter(e))

    print("finish")
