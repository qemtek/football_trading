from src.utils.db import connect_to_db, run_query


def get_upcoming_games(n=10):
    """Get the next n upcoming games in the Premier League"""
    conn, cursor = connect_to_db()
    query = "select * from fpl_fixtures where started = 0 order by kickoff_time limit {}".format(n)
    df = run_query(cursor, query)
    return df
