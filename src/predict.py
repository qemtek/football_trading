import pandas as pd

from src.models.MatchResultXGBoostModel import MatchResultXGBoost
from src.utils.api import get_upcoming_games
from src.update_tables import update_tables
from src.utils.db import run_query, connect_to_db


def predict():
    # Get the current season
    season = run_query(query='select max(season) from main_fixtures').iloc[0, 0]
    # Update tables
    update_tables()
    # Load the in-production model
    model = MatchResultXGBoost(load_trained_model=True, problem_name='match-predict-base')
    # Get upcoming games (that we haven't predicted)
    df = get_upcoming_games()
    # Make predictions
    predictions = pd.DataFrame(columns=['H', 'D', 'A'])
    for row in df.iterrows():
        input_dict = {
            "date": row[1]['kickoff_time'],
            "home_id": row[1]['home_id'],
            "away_id": row[1]['away_id'],
            "season": "19/20",
            "home_odds": row[1]['home_odds'],
            "draw_odds": row[1]['draw_odds'],
            "away_odds": row[1]['away_odds']
        }
        predictions_df = pd.DataFrame(model.predict(**input_dict), index=[len(predictions)])
        predictions = predictions.append(predictions_df)
    with connect_to_db() as conn:
        predictions.to_sql('latest_predictions', conn)


if __name__ == '__main__':
    predict()
