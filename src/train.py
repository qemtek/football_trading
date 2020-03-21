from src.models.MatchResultXGBoostModel import MatchResultXGBoost
from src.update_tables import update_tables


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
