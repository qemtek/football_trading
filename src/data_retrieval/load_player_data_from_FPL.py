from fpl import FPL
import aiohttp
import pandas as pd
from src.utils.db import connect_to_db, run_query
import asyncio


async def main():
    """Download all data from the Fantasy Premier League website"""
    # Connect to the sqlite3 DB
    conn, cursor = connect_to_db()
    summary_df = pd.DataFrame()

    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        players = await fpl.get_players()
        fixtures_by_gameweek = fpl.get_fixtures_by_gameweek()
        i=0
        for player in players:
            id = player.id
            i += 1
            print("Loading data for player {} of {}".format(i, len(players)))
            player_summary = await fpl.get_player_summary(id)
            player_history = pd.DataFrame(player_summary.history)
            summary_df = summary_df.append(player_history)

    # Upload the data to a table in the database
    run_query(cursor, 'DROP TABLE IF EXISTS player_data', return_data=False)
    summary_df.to_sql('player_data', conn)


if __name__ == '__main__':
    asyncio.run(main())
