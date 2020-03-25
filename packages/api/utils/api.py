from packages.football_trading.src.utils.db import run_query


def get_upcoming_games(n=10):
    """Get the next n upcoming games in the Premier League,
     with odds from the Betfair Exchange"""
    query = """select kickoff_time, t2.team_id home_id, t2.team_name home_name, 
    t3.team_id away_id, t3.team_name away_name, home_odds, draw_odds, away_odds
    from fpl_fixtures t1 
    left join fpl_teams t2 on t1.team_h = t2.id 
    left join fpl_teams t3 on t1.team_a = t3.id
    left join team_ids t4 on t4.team_id = t2.team_id
    left join bfex_latest_fixtures t5 on (
        date(t1.kickoff_time) = date(t5.market_start_time)
        and (t5.home_team = t4.team_name or 
             t5.home_team = t4.alternate_name or 
             t5.home_team = t4.alternate_name2))
    where started = 0 and home_odds is not NULL
    order by kickoff_time limit {}
    """.format(n)
    df = run_query(query=query)
    return df
