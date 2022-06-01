import pandas as pd

from src import data_dir

from sklearn.feature_extraction.text import CountVectorizer

class DataLoader:

    def __init__(self, dataset, task):
        #FIXME: add dataset path
        #self.dataset_path = data_dir / dataset
        self.dataset_path = data_dir
        self.task = task

    def _load_data(self):
        self.train = pd.read_csv(self.dataset_path / f'train_{self.task}.csv')
        self.test = pd.read_csv(self.dataset_path / f'test_{self.task}.csv')

    def _preprocess(self, data):
        #TODO: add preprocessing
        return data

    def _feature_embedding(self, train, test):
        embedder = CountVectorizer()
        X_train = embedder.fit_transform(train)
        X_test = embedder.transform(test)
        return X_train, X_test

    def load_and_preprocess(self):
        self._load_data()
        train, test = list(map(self._preprocess, [self.train, self.test]))
        X_train, X_test = self._feature_embedding(train['text'], test['text'])
        y_train, y_test = train['label'], test['label']
        return X_train, y_train, X_test, y_test
