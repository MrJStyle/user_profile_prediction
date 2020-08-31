import os

from pandas import DataFrame
from typing import List, Dict


current_file_path: str = os.path.abspath(__file__)


class BasePreprocess(object):
    data: DataFrame
    preprocess_data: DataFrame = DataFrame()

    split_word_sentences: List = list()

    dir_path: str = os.path.dirname(current_file_path)

    def __init__(self, file_path: str):
        self.data = self.load_from_csv(file_path)

    @classmethod
    def load_from_csv(cls, file_path: str) -> DataFrame: ...
    def count_word_freq(self) -> None: ...
