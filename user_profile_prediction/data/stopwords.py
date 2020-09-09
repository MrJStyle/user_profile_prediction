import os

import pandas as pd
from pandas import DataFrame

current_file_path: str = os.path.abspath(__file__)
dir_path: str = os.path.dirname(current_file_path)


class StopwordsDataset:
    stopwords: DataFrame

    def __init__(self, file_path: str = os.path.join(dir_path, "stopwords.txt")):
        self.stopwords = pd.read_csv(
            file_path,
            index_col=False,
            quoting=3,
            sep="\t",
            names=["stopword"],
            encoding="utf-8"
        )

    def __contains__(self, item):
        return item in self.stopwords.values


if __name__ == "__main__":
    s = StopwordsDataset()
    print("jjjjj" in s)
