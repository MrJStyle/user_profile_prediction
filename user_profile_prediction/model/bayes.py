import datetime

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from tensorflow.keras.callbacks import TensorBoard

from user_profile_prediction.etl.embedding import Embedding as E
from user_profile_prediction.etl.preprocess_train_data import PreprocessTrainingData


p: PreprocessTrainingData = PreprocessTrainingData(
    "/Volumes/Samsung_T5/Files/Document/china_hadoop/GroupProject/project_data/data/train.csv",
    embedding_size=200,
    sentence_len=5)
p.split_sentence()

e = E(200, 1)
m = e.train_word2vec_model(p.sentences_with_split_words)


x_train, x_val, y_train, y_val = p.split_data(p.age_data_iter(e), one_hot=False)

bayes = MultinomialNB()

bayes.fit(x_train, y_train)

predicted = bayes.predict(x_val)
print(classification_report(y_val, predicted))
print(confusion_matrix(y_val, predicted))