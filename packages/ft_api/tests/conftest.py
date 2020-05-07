import pytest

from ft_api.app import get_api
from ft_api.configuration import TestingConfig


@pytest.fixture
def app():
    app = get_api(config_object=TestingConfig)
    with app.app_context():
        yield app


@pytest.fixture
def flask_test_client(app):
    with app.test_client() as test_client:
        yield test_client
