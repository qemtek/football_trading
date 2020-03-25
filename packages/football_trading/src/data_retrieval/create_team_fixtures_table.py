from packages.football_trading.src.utils.db import connect_to_db


def create_team_fixtures():
    # Connect to database
    conn = connect_to_db()
    cursor = conn.cursor()
    # Calculate the home team stats
    cursor.execute("drop table if exists team_fixtures_home")
    cursor.execute("""CREATE TABLE team_fixtures_home as SELECT 
        fixture_id, 
        date, 
        home_team as team_name, 
        home_id as team_id,
        season, 
        home_score goals_for, 
        away_score goals_against, 
        home_score-away_score goal_difference,
        home_yellow_cards yellow_cards, 
        home_red_cards red_cards, 
        home_shots shots_for,
        away_shots shots_against,
        home_shots-away_shots shot_difference,
        1 as is_home,
        b365_home_odds as b365_win_odds,
        case 
            when home_score > away_score then 'W'
            when home_score < away_score then 'L' 
            when home_score = away_score then 'D' 
        end result,
        case when home_score > away_score then 3 
            when home_score < away_score then 0 
            when home_score = away_score then 1 
        end points
        FROM main_fixtures""")

    # Calculate the table for away only
    cursor.execute("drop table if exists team_fixtures_away")
    cursor.execute("""CREATE TABLE team_fixtures_away as SELECT 
        fixture_id, 
        date, 
        away_team as team_name, 
        away_id as team_id,
        season, 
        away_score goals_for, 
        home_score goals_against, 
        away_score-home_score goal_difference,
        away_yellow_cards yellow_cards, 
        away_red_cards red_cards, 
        away_shots shots_for,
        home_shots shots_against,
        away_shots-home_shots shot_difference,
        0 as is_home,
        b365_away_odds as b365_win_odds,
        case 
            when home_score > away_score then 'L'
            when home_score < away_score then 'W' 
            when home_score = away_score then 'D' 
        end result,
        case when home_score > away_score then 0 
            when home_score < away_score then 3 
            when home_score = away_score then 1 
        end points
        FROM main_fixtures""")

    # Combine the home and away table to get a full table
    cursor.execute("drop table if exists team_fixtures")
    cursor.execute("""CREATE TABLE team_fixtures as select * from team_fixtures_home 
    UNION ALL select * from team_fixtures_away""")
    conn.commit()
    conn.close()


if __name__ == '__main__':
    create_team_fixtures()
