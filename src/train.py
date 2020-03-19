from src.models.MatchResultXGBoostModel import MatchResultXGBoost

if __name__ == '__main__':
    new_model = MatchResultXGBoost(
        upload_historic_predictions=True,
        problem_name='match-predict',
        compare_models=True
    )
