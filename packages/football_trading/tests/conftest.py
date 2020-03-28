import os
import pandas as pd
import pytest

from packages.api.api_old import app
from packages.football_trading import PROJECTSPATH


@pytest.fixture
def test_dataset():
    test_data_dir = os.path.join(PROJECTSPATH, 'tests', 'test_data.csv')
    yield pd.read_csv(test_data_dir)


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client