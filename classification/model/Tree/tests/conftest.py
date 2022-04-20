import pytest
import pandas as pd
from pathlib import Path

path = Path(__file__)
data_dir = path.absolute().parent.parent.parent.parent / 'data'


@pytest.fixture(scope='session')
def bn_test_data():
    return pd.read_csv(data_dir / 'test_binary.csv')

@pytest.fixture(scope='session')
def bn_train_data():
    return pd.read_csv(data_dir / 'train_binary.csv')

@pytest.fixture(scope='session')
def accuracy_dict():
    return {
        'base_model': 0.5,
        'logistic_regression': 0.5,
    }