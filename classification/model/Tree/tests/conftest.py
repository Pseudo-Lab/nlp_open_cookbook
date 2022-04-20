import pytest
import pandas as pd
from pathlib import Path

path = Path(__file__)
data_dir = path.absolute().parent.parent.parent.parent / 'data'


@pytest.fixture
def bn_test_data():
    return pd.read_csv(data_dir / 'test_binary.csv')

