import re
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import List, Dict, Iterable, Tuple, Generator, Collection

import jieba
import numpy as np
import pandas as pd
from numpy import array
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tensorflow import Tensor, constant
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from user_profile_prediction.data.stopwords import StopwordsDataset
from user_profile_prediction.etl import BasePreprocess, EmbeddingModel

stop_words: StopwordsDataset = StopwordsDataset()


class PreprocessTrainingData(BasePreprocess):
    EMBEDDING_SIZE: int
    SENTENCE_LEN: int

    VOCABULARY_SIZE: int

    sample_num: int
    train_valid_test_weights: Collection

    age_label: List[int] = list()
    gender_label: List[int] = list()
    education_label: List[int] = list()

    age_label_weights: Dict[int, int]
    gender_label_weights: Dict[int, int]
    education_label_weights: Dict[int, int]

    tokenizer: Tokenizer

    saved_path: str = Path("../data/train_split_words.csv").absolute().__str__()

    @classmethod
    def load_from_csv(cls, file_path: str) -> DataFrame:
        df: DataFrame = pd.read_csv(file_path, sep="###__###", header=None)
        df.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']

        return df

    def __init__(
            self,
            csv_file_path: str,
            train_valid_weights: Collection = (0.9, 0.1),
            embedding_size: int = 100,
            sentence_len: int = 3,
            vocabulary_size: int = 10000
    ):
        super(PreprocessTrainingData, self).__init__(csv_file_path)
        self.preprocess_data.columns = list()
        self.train_valid_weights = train_valid_weights

        self.EMBEDDING_SIZE = embedding_size
        self.SENTENCE_LEN = sentence_len
        self.VOCABULARY_SIZE = vocabulary_size

    def split_sentence(self):
        indexes = self.data.index.to_list()
        # np.random.shuffle(indexes)

        saved_path: Path = Path(self.saved_path)
        if saved_path.exists():
            self.load_split_words_csv()
            return self.sentences_with_split_words

        for index, query in tqdm(self.data.iloc[indexes[:]].iterrows()):
            # TODO 测试模型由于计算资源有限，只用1000个样本做测试

            if query["Age"] == 0 or query["Gender"] == 0 or query["Education"] == 0:
                continue

            words = []

            for sentence in query["Query_List"].split("\t"):
                res = re.findall("(?isu)(https?\://[a-zA-Z0-9\.\?/&\=\:]+)", sentence)
                if not res:
                    cut_words: List = jieba.lcut(sentence)
                    cut_words = self.filter_stop_words(cut_words)

                    words.extend(cut_words)

            self.sentences_with_split_words.append(words)

            self.age_label.append(query["Age"])
            self.gender_label.append(query["Gender"])
            self.education_label.append(query["Education"])

        self.sample_num = len(self.sentences_with_split_words)

        self.age_label_weights = dict(Counter(self.age_label))
        self.gender_label_weights = dict(Counter(self.gender_label))
        self.education_label_weights = dict(Counter(self.education_label))

        self.save_split_words_csv()

        return self.sentences_with_split_words

    def save_split_words_csv(self) -> None:
        with open(self.saved_path, "w") as f:
            f.write("Query_List, Age, Gender, Education\n")

            for i in range(len(self.sentences_with_split_words)):
                f.write(
                    "{}, {}, {}, {}\n".format(
                        ' '.join(self.sentences_with_split_words[i]),
                        self.age_label[i],
                        self.gender_label[i],
                        self.education_label[i]
                    )
                )

    def load_split_words_csv(self) -> None:
        line: str

        with open(self.saved_path, "r") as f:
            for line in islice(f, 1, None):
                if line.endswith("\n"):
                    line = line[:-1]
                query_list, age, gender, edu = line.split(",")
                self.sentences_with_split_words.append(query_list.split(" "))
                self.age_label.append(int(age))
                self.gender_label.append(int(gender))
                self.education_label.append(int(edu))

    @property
    def train_valid_weights(self):
        return self._train_valid_weights

    @train_valid_weights.setter
    def train_valid_weights(self, weights: Collection):
        if weights.__len__() != 2:
            raise ValueError("set wrong dim weights")

        if sum(weights) != 1.:
            raise ValueError("sum of weights not equal to 1")

        self._train_valid_weights = weights

    @staticmethod
    def filter_stop_words(words: Iterable) -> List:
        return [w for w in words if w not in stop_words and w != " "]

    def age_data_iter(self) -> Generator[Tuple[array, int], None, None]:
        self.tokenizer: Tokenizer = Tokenizer(
            num_words=self.VOCABULARY_SIZE,
            oov_token="NaN",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.tokenizer.fit_on_texts(self.sentences_with_split_words)

        for i, s in enumerate(self.sentences_with_split_words):
            yield [x[0] if len(x) != 0 else 0 for x in self.tokenizer.texts_to_sequences(s)], self.age_label[i]

    def gender_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]:
        for i, s in enumerate(self.sentences_with_split_words):
            yield model.words_to_vec(s, self.SENTENCE_LEN), self.gender_label[i]

    def education_data_iter(self, model: EmbeddingModel) -> Generator[Tuple[array, int], None, None]:
        for i, s in enumerate(self.sentences_with_split_words):
            yield model.words_to_vec(s, self.SENTENCE_LEN), self.education_label[i]

    def split_data(self, data_iter: Generator, one_hot: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        all_data: List[Tuple[array, int]]
        all_data = [(x, y) for x, y in data_iter]

        x_train_raw, y_train_raw = zip(*all_data)
        x_train_raw = pad_sequences(
                x_train_raw,
                maxlen=self.SENTENCE_LEN,
                padding="post",
                truncating="post"
            )

        y_train_raw = array(y_train_raw)
        # x_train_raw, y_train_raw = \
        #     np.stack(x_train_raw, axis=0).reshape(-1, self.SENTENCE_LEN * self.EMBEDDING_SIZE), array(y_train_raw)

        x_train: array
        x_val: array
        y_train: array
        y_val: array

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_raw, y_train_raw,
            test_size=self._train_valid_weights[1],
            random_state=101
        )

        # ros: RandomOverSampler = RandomOverSampler(random_state=202)
        # x_train, y_train = ros.fit_resample(x_train, y_train)

        # x_train: Tensor = constant(x_train.reshape(-1, self.SENTENCE_LEN, self.EMBEDDING_SIZE))
        # x_val: Tensor = constant(x_val.reshape(-1, self.SENTENCE_LEN, self.EMBEDDING_SIZE))
        x_train: Tensor = constant(x_train)
        x_val: Tensor = constant(x_val)
        if one_hot:
            y_train: Tensor = tf.one_hot(y_train, depth=np.unique(y_train_raw).__len__())
            y_val: Tensor = tf.one_hot(y_val, depth=np.unique(y_train_raw).__len__())

        return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    import tensorflow as tf

    p = PreprocessTrainingData("/Volumes/Samsung_T5/Files/Document/china_hadoop/GroupProject/project_data/data/train.csv")
    p.split_sentence()
    # a, b, c, d = p.split_data(p.age_data_iter())

    print("finish")
