import pytest

from ft_api.app import get_api
from ft_api.settings import TestingConfig


@pytest.fixture
def app():
    app = get_api(input_config=TestingConfig)
    with app.app_context():
        yield app


@pytest.fixture
def flask_test_client(app):
    with app.test_client() as test_client:
        yield test_client
