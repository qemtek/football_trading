from fpl import FPL
import aiohttp
import pandas as pd
import asyncio

from football_trading.src.utils.db import connect_to_db, run_query


async def get_fpl_fixtures():
    """Download upcoming fixture data from the Fantasy Premier League website"""
    # Connect to the sqlite3 DB
    conn = connect_to_db()
    fixture_list = pd.DataFrame()

    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        fixtures = await fpl.get_fixtures(return_json=True)
    for fixture in fixtures:
        del fixture['stats']
        fixture_list = fixture_list.append(pd.DataFrame(fixture, index=[0]))

    # Upload the data to a table in the database
    run_query(query='DROP TABLE IF EXISTS fpl_fixtures', return_data=False)
    fixture_list.to_sql('fpl_fixtures', conn)


if __name__ == '__main__':
    asyncio.run(get_fpl_fixtures())
