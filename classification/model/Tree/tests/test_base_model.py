import pytest

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class BaseModel:
    def __init__(self):
        pass

    def predict(self, data):
        return '긍정'


@pytest.fixture(scope='session')
def X_data_emb(bn_train_data, bn_test_data):
    train_x = bn_train_data['text']
    test_x = bn_test_data['text']

    bow = CountVectorizer()
    train_x_emb = bow.fit_transform(train_x)
    test_x_emb = bow.transform(test_x)
    return train_x_emb, test_x_emb

class TestModelAccuracy:
    
    def test_base_model(self, test_x, test_y, accuracy_dict):
        base_model = BaseModel()
        predict = list(map(base_model.predict, test_x))
        accuracy = sum(predict == test_y) / len(test_y)
        accuracy_dict['base_model'] = accuracy
        print("BaseModel accuracy:", accuracy)
        assert accuracy >= 0.5

    def test_logistic_regression(self, train_y, test_y, X_data_emb, accuracy_dict):
        train_x_emb, test_x_emb = X_data_emb
        lr = LogisticRegression()
        lr.fit(train_x_emb, train_y)
        predict = lr.predict(test_x_emb)
        accuracy = sum(predict == test_y) / len(test_y)
        accuracy_dict['logistic_regression'] = accuracy
        print("LogisticRegression accuracy:", accuracy) 
        assert accuracy >= accuracy_dict['base_model']  # 0.7665

    def test_decison_tree(self, train_y, test_y, X_data_emb, accuracy_dict):
        train_x_emb, test_x_emb = X_data_emb
        dt = DecisionTreeClassifier()
        dt.fit(train_x_emb, train_y)
        predict = dt.predict(test_x_emb)
        accuracy = sum(predict == test_y) / len(test_y)
        print("DecisionTree accuracy:", accuracy) 
        assert accuracy >= accuracy_dict['logistic_regression']

    def test_random_forest(self,  train_y, test_y, X_data_emb, accuracy_dict):
        train_x_emb, test_x_emb = X_data_emb
        rf = RandomForestClassifier()
        rf.fit(train_x_emb, train_y)
        predict = rf.predict(test_x_emb)
        accuracy = sum(predict == test_y) / len(test_y)
        print("RandomForest accuracy:", accuracy) 
        assert accuracy >= accuracy_dict['logistic_regression']
