import asyncio
import logging

from src.data_retrieval.load_fixture_list_from_FPL import get_fpl_fixtures
from src.data_retrieval.load_fixures_from_fbd import update_fixtures_from_fbd
from src.data_retrieval.create_team_fixtures_table import create_team_fixtures
from src.data_retrieval.load_manager_data_from_wiki import get_manager_data
from src.data_retrieval.load_player_data_from_FPL import get_player_data
from src.data_retrieval.load_PL_teams_from_wiki import get_teams_from_wiki

logger = logging.getLogger('API')


def update_tables():
    # Get all historic premier league teams from Wikipedia
    get_teams_from_wiki()
    logger.info('get_teams_from_wiki complete [1/6]')
    # Download latest fixture data from football-data.co.uk
    update_fixtures_from_fbd()
    logger.info('update_fixtures_from_fbd complete [2/6]')
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


if __name__ == '__main__':
    update_tables()
