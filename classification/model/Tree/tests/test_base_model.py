import pytest

class BaseModel:
    def __init__(self):
        pass

    def predict(self, data):
        return '긍정'

def test_base_model(bn_test_data):
    test_x = bn_test_data['text']
    test_y = bn_test_data['label']

    base_model = BaseModel()
    predict = list(map(base_model.predict, test_x))
    accuracy = sum(predict == test_y) / len(test_y)
    assert accuracy >= 0.5