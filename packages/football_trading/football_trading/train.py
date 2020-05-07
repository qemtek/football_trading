from football_trading.src.models.MatchResultXGBoostModel import MatchResultXGBoost
from football_trading.update_tables import update_tables
from football_trading.src.utils.logging import get_logger
from football_trading.settings import LOCAL

logger = get_logger()


def train_new_model(problem_name='match-predict-base', test_mode=False):
    # Update tables
    update_tables()
    # Train new model
    MatchResultXGBoost(
        upload_historic_predictions=True,
        problem_name=problem_name,
        production_model=True,
        compare_models=True,
        test_mode=test_mode)


if __name__ == '__main__':
    train_new_model()
