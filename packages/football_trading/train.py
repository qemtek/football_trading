from src.models.MatchResultXGBoostModel import MatchResultXGBoost
from src.update_tables import update_tables
from src.utils.logging import get_logger

logger = get_logger()


def train_new_model(problem_name='match_predict_base'):
    # Update tables
    update_tables()
    # Train new model
    MatchResultXGBoost(
        upload_historic_predictions=True,
        problem_name=problem_name,
        production_model=True,
        compare_models=True)


if __name__ == '__main__':
    train_new_model()
