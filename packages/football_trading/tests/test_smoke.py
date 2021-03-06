import os

from football_trading.settings import PROJECTSPATH
from football_trading.train import train_new_model


def test_train_model():
    try:
        os.environ['DB_DIR'] = f"{os.path.dirname(PROJECTSPATH)}/tests/'test_db.sqlite"
        train_new_model(test_mode=True)
    except Exception as e:
        assert False, f'Failed smoke testing train_model, error: {e}'