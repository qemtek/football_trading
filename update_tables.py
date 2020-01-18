from src.data_retrieval.load_fixture_list_from_FPL import get_fpl_fixtures
from src.data_retrieval.load_fixures_from_fbd import update_fixtures_from_fbd
from src.data_retrieval.create_team_fixtures_table import create_team_fixtures
from src.data_retrieval.load_manager_data_from_wiki import get_manager_data
from src.data_retrieval.load_player_data_from_FPL import get_player_data
from src.data_retrieval.load_PL_teams_from_wiki import get_teams_from_wiki

# Get all historic premier league teams from Wikipedia
get_teams_from_wiki()
# Download latest fixture data from football-data.co.uk
update_fixtures_from_fbd()
# Create team_fixtures table (aggregated from fbd info
create_team_fixtures()
# Get manager data from Wikipedia
get_manager_data()
# Get player-level data from the Fantasy Premier League website
get_player_data()
# Get upcoming fixtures from FPL page
get_fpl_fixtures()
