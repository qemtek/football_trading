import asyncio
import datetime as dt
import os

from src.utils.db import run_query
from src.data_retrieval.load_fixture_list_from_FPL import get_fpl_fixtures
from src.data_retrieval.load_fixures_from_fbd import update_fixtures_from_fbd
from src.data_retrieval.create_team_fixtures_table import create_team_fixtures
from src.data_retrieval.load_manager_data_from_wiki import get_manager_data
from src.data_retrieval.load_player_data_from_FPL import get_player_data
from src.data_retrieval.load_PL_teams_from_wiki import get_teams_from_wiki
from src.data_retrieval.get_latest_fixtures_from_bfex import get_latest_fixtures
from src.utils.logging import get_logger
from settings import db_dir, RECREATE_DB

logger = get_logger()


def update_tables(*, recreate_table=RECREATE_DB) -> None:
    """Update all tables in the src/data_retrieval folder"""
    # If we are recreating the DB, delete any existing db
    if recreate_table:
        if os.path.exists(db_dir):
            os.rmdir(db_dir)
        unrecorded_games = [1]
    else:
        try:
            # Get the last fixture in the main_fixtures table
            last_game = run_query(query='select max(date) from main_fixtures').iloc[0, 0]
            # Get the current date
            today = dt.datetime.today().date()
            # Find out how many fixtures have taken place between the last fixture and the current date
            unrecorded_games = run_query(query=f"select * from fpl_fixtures where date(kickoff_time) "
                                               f"between '{last_game}' and '{today}'")
        except Exception as e:
            raise Exception(f'It is likely that the db at {db_dir} could not be found. Original error: {e}')
    # Update tables if there are games we have not logged in the DB
    if len(unrecorded_games) > 0:
        # Download latest fixture data from football-data.co.uk
        update_fixtures_from_fbd()
        logger.info('update_fixtures_from_fbd complete [1/6]')
        # Get all historic premier league teams from Wikipedia
        get_teams_from_wiki()
        logger.info('get_teams_from_wiki complete [2/6]')
        # Create team_fixtures table (aggregated from fbd info
        create_team_fixtures()
        logger.info('create_team_fixtures complete [3/6]')
        # Get manager data from Wikipedia
        get_manager_data()
        logger.info('get_manager_data complete [4/6]')
        # Get player-level data from the Fantasy Premier League website
        asyncio.run(get_player_data())
        logger.info('get_player_data complete [5/6]')
        # Get upcoming fixtures from FPL page
        asyncio.run(get_fpl_fixtures())
        logger.info('get_fpl_fixtures complete [6/6]')
        # Get latest odds from the Betfair Exchange API
        get_latest_fixtures()
    else:
        logger.info('Tables are up to date, skipping update.')


if __name__ == '__main__':
    update_tables()
