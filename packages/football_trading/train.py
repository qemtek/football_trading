from packages.football_trading.src.models.MatchResultXGBoostModel import MatchResultXGBoost
from packages.football_trading.src.update_tables import update_tables
from packages.football_trading.logging_config import get_logger

logger = get_logger()


def train():
    # Update tables
    update_tables()
    # Train new model
    MatchResultXGBoost(
        upload_historic_predictions=True,
        problem_name='match-predict-base',
        production_model=True,
        compare_models=True)


if __name__ == '__main__':
    train()
