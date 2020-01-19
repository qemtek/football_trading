from src.utils.db import connect_to_db, run_query


def get_upcoming_games(n=10):
    """Get the next n upcoming games in the Premier League"""
    conn, cursor = connect_to_db()
    query = """select kickoff_time, t2.team_id home_id, t2.team_name home_name, 
    t3.team_id away_id, t3.team_name away_name
    from fpl_fixtures t1 left join fpl_teams t2 on t1.team_h = t2.id left 
    join fpl_teams t3 on t1.team_a = t3.id where started = 0 order by 
    kickoff_time limit {}""".format(n)
    df = run_query(cursor, query)
    return df
