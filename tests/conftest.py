import os
import pandas as pd
import pytest

from src.api import app
from configuration import project_dir


@pytest.fixture
def test_dataset():
    test_data_dir = os.path.join(project_dir, 'tests', 'test_data.csv')
    yield pd.read_csv(test_data_dir)


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client