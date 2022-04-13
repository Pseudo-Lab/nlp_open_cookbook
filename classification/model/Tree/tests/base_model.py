import pytest

class BaseModel:
    def __init__(self):
        pass

    def predict(self, data):
        return 1

def test_base_model():
    data = 'fake_data'
    base_model = BaseModel()
    assert base_model.predict(data) == 1