from src.models.MatchResultXGBoostModel import MatchResultXGBoost

if __name__ == '__main__':
    model = MatchResultXGBoost(
        load_trained_model=True,
        problem_name='match-predict'
    )
    # Get upcoming games (that we havent predicted)

    # Make predictions
