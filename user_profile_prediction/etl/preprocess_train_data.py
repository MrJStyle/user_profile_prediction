import os
import jieba
import pandas as pd

from typing import List, Iterable
from pandas import DataFrame
from collections import Counter
from gensim.models import Word2Vec

from user_profile_prediction.etl import BasePreprocess
from user_profile_prediction.data.stopwords import StopwordsDataset

from tqdm import tqdm


stop_words: StopwordsDataset = StopwordsDataset()


class PreprocessTrain(BasePreprocess):
    embedding_model: Word2Vec

    MIN_COUNT: int = 5
    SIZE: int = 100
    EMBEDDING_MODEL_SAVED_PATH: str = os.path.join(BasePreprocess.dir_path, "embedding_model.txt")

    age_label: List[str] = list()
    gender_label: List[str] = list()
    education_label: List[str] = list()

    @classmethod
    def load_from_csv(cls, file_path: str) -> DataFrame:
        df: DataFrame = pd.read_csv(file_path, sep="###__###", header=None)
        df.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']

        return df

    def __init__(self, file_path: str):
        super(PreprocessTrain, self).__init__(file_path)
        self.preprocess_data.columns = []

    def train_word2vec_model(self) -> Word2Vec:
        for index, query in tqdm(self.data.iterrows()):
            if index == 1000:
                break

            for sentence in query["Query_List"].split("\t"):
                self.age_label.append(query["Age"])
                self.gender_label.append(query["Gender"])
                self.education_label.append(query["Education"])

                cut_words: List = jieba.lcut(sentence)
                self.split_word_sentences.append(self.filter_stop_words(cut_words))

        self.embedding_model: Word2Vec = Word2Vec(self.split_word_sentences, min_count=self.MIN_COUNT, size=self.SIZE)

        if os.path.exists(self.EMBEDDING_MODEL_SAVED_PATH):
            os.remove(self.EMBEDDING_MODEL_SAVED_PATH)

        self.embedding_model.save(self.EMBEDDING_MODEL_SAVED_PATH)

        return self.embedding_model

    @staticmethod
    def filter_stop_words(words: Iterable) -> List:
        return [w for w in words if w not in stop_words and w != " "]


if __name__ == "__main__":
    p = PreprocessTrain("/Volumes/Samsung_T5/Files/Document/小象学院/GroupProject/project_data/data/train.csv")
    model = p.train_word2vec_model()
    print("finish")
