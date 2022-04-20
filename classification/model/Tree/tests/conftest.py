import pytest
import pandas as pd
from pathlib import Path

path = Path(__file__)
data_dir = path.absolute().parent.parent.parent.parent / 'data'


@pytest.fixture(scope='session')
def bn_test_data():
    return pd.read_csv(data_dir / 'test_binary.csv')

def train_x(bn_train_data):
    return bn_train_data['text']

def train_y(bn_train_data):
    return bn_train_data['label']

@pytest.fixture(scope='session')
def bn_train_data():
    return pd.read_csv(data_dir / 'train_binary.csv')

def test_x(bn_test_data):
    return bn_test_data['text']

def test_y(bn_test_data):
    return bn_test_data['label']

@pytest.fixture(scope='session')
def accuracy_dict():
    return {
        'base_model': 0.5,
        'logistic_regression': 0.5,
    }