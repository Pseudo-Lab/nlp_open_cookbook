import pytest

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class BaseModel:
    def __init__(self):
        pass

    def predict(self, data):
        return '긍정'


class TestBaseModel:
    
    def test_base_model(self, bn_test_data, accuracy_dict):
        test_x = bn_test_data['text']
        test_y = bn_test_data['label']

        base_model = BaseModel()
        predict = list(map(base_model.predict, test_x))
        accuracy = sum(predict == test_y) / len(test_y)
        accuracy_dict['base_model'] = accuracy
        print("BaseModel accuracy:", accuracy)
        assert accuracy >= 0.5

    def test_logistic_regression(self, bn_train_data, bn_test_data, accuracy_dict):
        train_x = bn_train_data['text']
        train_y = bn_train_data['label']

        test_x = bn_test_data['text']
        test_y = bn_test_data['label']

        bow = CountVectorizer()
        train_x_emb = bow.fit_transform(train_x)
        test_x_emb = bow.transform(test_x)

        lr = LogisticRegression()
        lr.fit(train_x_emb, train_y)
        predict = lr.predict(test_x_emb)
        accuracy = sum(predict == test_y) / len(test_y)
        accuracy_dict['logistic_regression'] = accuracy
        print("LogisticRegression accuracy:", accuracy) 
        assert accuracy >= accuracy_dict['base_model']  # 0.7665

    def test_decison_tree(self, bn_train_data, bn_test_data, accuracy_dict):
        train_x = bn_train_data['text']
        train_y = bn_train_data['label']

        test_x = bn_test_data['text']
        test_y = bn_test_data['label']

        bow = CountVectorizer()
        train_x_emb = bow.fit_transform(train_x)
        test_x_emb = bow.transform(test_x)

        dt = DecisionTreeClassifier()
        dt.fit(train_x_emb, train_y)
        predict = dt.predict(test_x_emb)
        accuracy = sum(predict == test_y) / len(test_y)
        print("DecisionTree accuracy:", accuracy) 
        assert accuracy >= accuracy_dict['logistic_regression']
