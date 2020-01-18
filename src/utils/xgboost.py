import numpy as np
import pandas as pd
import logging
from src.utils.db import connect_to_db, run_query

logger = logging.getLogger('XGBoostModel')


def get_performance_vs_bookmaker(df):
    """Creates a score relating to how much the team beats
    (or doesn't beat) bookmaker expectations"""

    def calculate_score(row):
        return 1 / row['b365_win_odds'] if row['result'] == 'W' \
            else 1 - 1 / row['b365_win_odds']

    df['perf_vs_bm'] = df.apply(lambda x: calculate_score(x), axis=1)
    return sum(df['perf_vs_bm'])


def get_home_away_advantage(df, type):
    df['goal_dif'] = df['goals_for'] - df['goals_against']
    home_data = df.loc[df['is_home'] == 1, :]
    away_data = df.loc[df['is_home'] == 0, :]
    home_goal_dif_sum = sum(home_data['goal_dif'])
    home_goal_dif_avg = np.mean(home_data['goal_dif'])
    away_goal_dif_sum = sum(away_data['goal_dif'])
    away_goal_dif_avg = np.mean(away_data['goal_dif'])
    home_advantage_sum = (0.1234 + home_goal_dif_sum) / (0.1234 + away_goal_dif_sum)
    home_advantage_avg = (0.1234 + home_goal_dif_avg) / (0.1234 + away_goal_dif_avg)
    return [home_advantage_sum, home_advantage_avg] if type == 'home' \
        else [1 / home_advantage_sum, 1 / home_advantage_avg]


def get_features(row, index, team_data, window_length=8, type='home'):
    # ToDo: Use team ID instead of team name
    team_id = row['home_id' if type == 'home' else 'away_id']
    fixture_id = row['fixture_id']
    season = row['season']
    # Filter for the team/season
    df_filtered = team_data[(team_data['team_id'] == team_id) &
                      (team_data['season'] == season) &
                      (team_data['fixture_id'] < fixture_id)]
    # Get the last 8 games
    df_filtered = df_filtered.sort_values('date').tail(window_length)
    # Create aggregated features
    df_output = pd.DataFrame()
    df_output.loc[index, 'avg_goals_for_' + type] = np.mean(df_filtered['goals_for'])
    df_output.loc[index, 'avg_goals_against_' + type] = np.mean(
        df_filtered['goals_against'])
    df_output.loc[index, 'sd_goals_for_' + type] = np.std(df_filtered['goals_for'])
    df_output.loc[index, 'sd_goals_against_' + type] = np.std(df_filtered['goals_against'])
    df_output.loc[index, 'avg_shots_for_' + type] = np.mean(df_filtered['shots_for'])
    df_output.loc[index, 'avg_shots_against_' + type] = np.mean(
        df_filtered['shots_against'])
    df_output.loc[index, 'sd_shots_for_' + type] = np.std(df_filtered['shots_for'])
    df_output.loc[index, 'sd_shots_against_' + type] = np.std(df_filtered['shots_against'])
    df_output.loc[index, 'avg_yellow_cards_' + type] = np.mean(df_filtered['yellow_cards'])
    df_output.loc[index, 'avg_red_cards_' + type] = np.mean(df_filtered['red_cards'])
    df_output.loc[index, 'b365_win_odds_' + type] = np.mean(df_filtered['b365_win_odds'])
    df_output.loc[index, 'avg_perf_vs_bm_' + type] = get_performance_vs_bookmaker(
        df_filtered)
    df_output.loc[index, 'manager_new_' + type] = row[type + '_manager_new']
    df_output.loc[index, 'manager_age_' + type] = row[type + '_manager_age']
    df_output.loc[index, 'win_rate_' + type] = np.mean(
        df_filtered['result'].apply(lambda x: 1 if x == 'W' else 0))
    df_output.loc[index, 'draw_rate_' + type] = np.mean(
        df_filtered['result'].apply(lambda x: 1 if x == 'D' else 0))
    df_output.loc[index, 'loss_rate_' + type] = np.mean(
        df_filtered['result'].apply(lambda x: 1 if x == 'L' else 0))
    ha_features = get_home_away_advantage(df_filtered, type)
    df_output.loc[index, 'home_advantage_sum_' + type] = ha_features[0]
    df_output.loc[index, 'home_advantage_avg_' + type] = ha_features[1]
    return df_output


def calculate_win_streak(last_games):
    count = 0
    while last_games.iloc[count] == 1:
        count += 1
    return count


def get_manager(team_id, date):
    """Find the sitting manager for a given team_id and date"""
    query = """select * from managers where team_id = {} 
            and start < '{}' and end >= '{}'""".format(team_id, date, date)
    conn, cursor = connect_to_db()
    df = run_query(cursor, query)
    rows = len(df)
    conn.close()
    if rows != 1:
        logger.warning("get_manager: Expected 1 row but got {}. Is the manager "
                       "info up to date?".format(rows))
    return df


def get_manager_features(df):
    """Get manager features (time as manager) (manager age is logged to reduce scale)"""
    df['date'] = pd.to_datetime(df['date'])
    df['home_manager_start'] = pd.to_datetime(df['home_manager_start'])
    df['home_manager_age'] = df.apply(
        lambda x: np.log10(round((x['date'] - x['home_manager_start']).days)), axis=1)
    df['away_manager_start'] = pd.to_datetime(df['away_manager_start'])
    df['away_manager_age'] = df.apply(
        lambda x: np.log10(round((x['date'] - x['away_manager_start']).days)), axis=1)
    df['home_manager_new'] = df['home_manager_age'].apply(lambda x: 1 if x <= 70 else 0)
    df['away_manager_new'] = df['away_manager_age'].apply(lambda x: 1 if x <= 70 else 0)
    return df


def get_feature_data(min_training_data_date='2013-08-01'):
    conn, cursor = connect_to_db()
    df = run_query(cursor, """select t1.*, m_h.manager home_manager,
     m_h.start home_manager_start, 
     m_a.manager away_manager, m_a.start away_manager_start 
     from main_fixtures t1 
     left join managers m_h 
     on t1.home_id = m_h.team_id 
     and (t1.date between m_h.start and date(m_h.end, '+1 day') 
     or t1.date > m_h.start and m_h.end is NULL) 
     left join managers m_a 
     on t1.away_id = m_a.team_id 
     and (t1.date between m_a.start and date(m_a.end, '+1 day') 
     or t1.date > m_a.start and m_a.end is NULL) 
     where t1.date > '{}'""".format(min_training_data_date))
    df=get_manager_features(df)
    df2 = run_query(cursor, "select * from team_fixtures where date > '{}'".format(
        min_training_data_date))
    conn.close()
    df2['date'] = pd.to_datetime(df2['date'])
    df2 = pd.merge(
        df2,
        df[['date', 'season', 'fixture_id', 'home_manager_age', 'away_manager_age',
            'home_manager_new', 'away_manager_new']],
        on=['date', 'season', 'fixture_id'],
        how="left")
    return df2