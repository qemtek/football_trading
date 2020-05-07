import pandas as pd
import datetime as dt

from settings import LOCAL
from src.models.MatchResultXGBoostModel import MatchResultXGBoost
from src.utils.api import get_upcoming_games
from update_tables import update_tables
from src.utils.db import run_query, connect_to_db
from src.utils.logging import get_logger
from football_trading import __version__

logger = get_logger()


def make_predictions():
    # Update tables
    update_tables()
    # Get the current season
    current_season = run_query(query='select max(season) from main_fixtures').iloc[0, 0]
    # Load the in-production model
    model = MatchResultXGBoost(load_trained_model=True, problem_name='match-predict-base', local=False)
    # Get upcoming games (that we haven't predicted)
    df = get_upcoming_games()
    logger.info(f'Making predictions for upcoming games: {df}')
    # Make predictions
    predictions_df = pd.DataFrame(columns=['date', 'home_id', 'away_id', 'season', 'home_odds',
                                           'draw_odds', 'away_odds', 'H', 'D', 'A', 'model', 'version',
                                           'creation_time'])
    for row in df.iterrows():
        input_dict = {
            "date": row[1]['kickoff_time'],
            "home_id": row[1]['home_id'],
            "away_id": row[1]['away_id'],
            "season": current_season,
            "home_odds": row[1]['home_odds'],
            "draw_odds": row[1]['draw_odds'],
            "away_odds": row[1]['away_odds'],
        }
        # Get predictions from model
        predictions = model.predict(**input_dict)
        # Combine predictions with input data
        output_dict = dict(input_dict, **predictions)
        # Convert dictionary to DataFrame
        output_df = pd.DataFrame(output_dict, columns=predictions_df.columns, index=len(predictions_df))
        # Add row to output DataFrame
        predictions_df = predictions_df.append(output_df)
        predictions_df["model"] = model.model_id,
        predictions_df["version"] = __version__,
        predictions_df["creation_time"] = str(dt.datetime.today())
    with connect_to_db() as conn:
        # Upload output DataFrame to DB
        predictions_df.to_sql('latest_predictions', conn, if_exists='replace')


if __name__ == '__main__':
    make_predictions()
