import pandas as pd
import pytest

from football_trading.settings import PROJECTSPATH


@pytest.fixture
def test_dataset():
    test_data_dir = f"{PROJECTSPATH}/tests/test_data.csv"
    yield pd.read_csv(test_data_dir)
