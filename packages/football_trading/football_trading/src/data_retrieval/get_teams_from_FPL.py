from fpl import FPL
import aiohttp
import pandas as pd
import asyncio

from football_trading.src.utils.team_id_functions import find_closest_match
from football_trading.src.utils.db import connect_to_db, run_query


async def get_fpl_teams():
    """Download upcoming fixture data from the Fantasy Premier League website"""
    # Connect to the sqlite3 DB
    conn = connect_to_db()
    df_teams = pd.DataFrame()
    i=0
    # Get all of the team IDs from the fpl_fixtures table
    teams = run_query('select distinct team_h from fpl_fixtures')
    n_teams = len(teams)
    assert n_teams == 20, 'The number of returned teams should be 20, ' \
                          'but is actually {}'.format(n_teams)
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        team_data = await fpl.get_teams(return_json=True)

    for team in team_data:
        df_teams = df_teams.append(pd.DataFrame(team, index=[i]))
        i += 1
    # Find the closest matching team names to the ones in our data
    matches = list(df_teams['name'].apply(lambda x: find_closest_match(x)))
    df_matches = pd.DataFrame()
    for match in matches:
        df_matches = df_matches.append(pd.DataFrame(match, index=[0]))
    df_teams = pd.concat([df_teams, df_matches.reset_index(drop=True)], axis=1)
    # Upload the data to a table in the database
    run_query('DROP TABLE IF EXISTS fpl_teams', return_data=False)
    df_teams.to_sql('fpl_teams', conn)
    conn.close()


if __name__ == '__main__':
    test = asyncio.run(get_fpl_teams())
