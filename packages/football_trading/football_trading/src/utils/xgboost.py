import numpy as np
import pandas as pd
import logging

from football_trading.src.utils.db import run_query

logger = logging.getLogger()


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


def get_features(row, index, team_data, identifiers, window_length=8):
    """Generate features for a given fixture"""
    # Generate features for the home team
    home_features = get_features_ha(
        row,
        team_data=team_data,
        type='home',
        window_length=window_length,
        index=index)
    # Generate features for the away team
    away_features = get_features_ha(
        row,
        team_data=team_data,
        type='away',
        window_length=window_length,
        index=index)
    # Combine identifiers, home features and away features
    odds_cols = ['home_odds', 'draw_odds', 'away_odds']
    row_cols = identifiers + odds_cols
    features = pd.concat([
        pd.DataFrame(row[row_cols].to_dict(), columns=row_cols, index=[index]),
        home_features, away_features], axis=1)
    return features


def get_features_ha(row, index, team_data, window_length=8, type='home'):
    """Generate features for the home team or away team"""
    team_id = row['home_id' if type == 'home' else 'away_id']
    fixture_id = row['fixture_id']
    season = row['season']
    # Filter for the team/season
    df_filtered = team_data[(team_data['team_id'] == team_id) &
                      (team_data['season'] == season) &
                      (team_data['fixture_id'] < fixture_id)]

    # Get the last 8 games
    df_filtered = df_filtered.sort_values('date').tail(window_length).reset_index()
    # Create aggregated features
    df_output = pd.DataFrame()
    df_output.loc[index, 'avg_goals_for_' + type] = np.mean(df_filtered['goals_for'])
    df_output.loc[index, 'avg_goals_against_' + type] = np.mean(
        df_filtered['goals_against'])
    df_output.loc[index, 'avg_goals_for_ha_' + type] = np.mean(
        df_filtered[df_filtered['is_home'] == (1 if type == 'home' else 0)]['goals_for'])
    df_output.loc[index, 'avg_goals_against_ha_' + type] = np.mean(
        df_filtered[df_filtered['is_home'] == (1 if type == 'home' else 0)]['goals_against'])
    df_output.loc[index, 'sd_goals_for_' + type] = np.std(df_filtered['goals_for'])
    df_output.loc[index, 'sd_goals_against_' + type] = np.std(df_filtered['goals_against'])
    df_output.loc[index, 'avg_shots_for_' + type] = np.mean(df_filtered['shots_for'])
    df_output.loc[index, 'avg_shots_against_' + type] = np.mean(df_filtered['shots_against'])
    df_output.loc[index, 'sd_shots_for_' + type] = np.std(df_filtered['shots_for'])
    df_output.loc[index, 'sd_shots_against_' + type] = np.std(df_filtered['shots_against'])
    df_output.loc[index, 'avg_yellow_cards_' + type] = np.mean(df_filtered['yellow_cards'])
    df_output.loc[index, 'avg_red_cards_' + type] = np.mean(df_filtered['red_cards'])
    df_output.loc[index, 'avg_perf_vs_bm_' + type] = get_performance_vs_bookmaker(df_filtered)
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
    # ToDo: Add win streak

    # ToDo: Add game level metrics for the last 5 games
    for i in range(1, 6):
        df_output.loc[index, f'is_home_l{i}_{type}'] = df_filtered.loc[len(df_filtered)-i, 'is_home']
        df_output.loc[index, f'goals_for_l{i}_{type}'] = df_filtered.loc[len(df_filtered)-i, 'goals_for']
        df_output.loc[index, f'goals_against_l{i}_{type}'] = df_filtered.loc[len(df_filtered)-i, 'goals_against']
        df_output.loc[index, f'goal_difference_l{i}_{type}'] = df_output.loc[index, f'goals_for_l{i}_{type}'] - \
                                                         df_output.loc[index, f'goals_against_l{i}_{type}']
        df_output.loc[index, f'shots_for_l{i}_{type}'] = df_filtered.loc[len(df_filtered)-i, 'goals_for']
        df_output.loc[index, f'shots_against_l{i}_{type}'] = df_filtered.loc[len(df_filtered)-i, 'shots_against']
        df_output.loc[index, f'shot_difference_l{i}_{type}'] = df_output.loc[index, f'shots_for_l{i}_{type}'] - \
                                                         df_output.loc[index, f'shots_against_l{i}_{type}']
    return df_output


def calculate_win_streak(last_games):
    count = 0
    while last_games.iloc[count] == 1:
        count += 1
    return count


def get_manager(team_id, date):
    """Find the sitting manager for a given team_id and date"""
    # Get the latest date in the db
    max_date = run_query(query="select max(end_date) date from managers").loc[0, 'date']
    # If we are predicting past our data
    if date > max_date:
        # Take the latest manager for the team
        query = """select * from managers where end_date = '{}' and 
        team_id = {}""".format(max_date, team_id)
    else:
        query = """select * from managers where team_id = {} 
                    and '{}' between start_date and end_date""".format(team_id, date)
    df = run_query(query=query)
    rows = len(df)
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
    # Convert any -inf to 0 (this occurs when we do np.log10(-1)
    df = df.replace([np.inf, -np.inf], 0)
    return df


def get_profit(x):
    if x['pred'] == x['actual']:
        if x['actual'] == 'H':
            return x['home_odds'] - 1
        elif x['actual'] == 'D':
            return x['draw_odds'] - 1
        elif x['actual'] == 'A':
            return x['away_odds'] - 1
        else:
            raise Exception('full_time_result is not H, D or A.')
    else:
        return -1


def get_profit2(x):
    if x['pred'] == x['actual']:
        if x['actual'] == 1:
            if x['model1_actual'] == 'H':
                return x['home_odds'] - 1
            elif x['model1_actual'] == 'D':
                return x['draw_odds'] - 1
            elif x['model1_actual'] == 'A':
                return x['away_odds'] - 1
            else:
                raise Exception('full_time_result is not H, D or A.')
        elif x['actual'] == 0:
            return 0
        else:
            raise Exception('get_profit2: prediction is neither 1 nor 0.')
    else:
        if x['actual'] == 1:
            return 0
        elif x['actual'] == 0:
            return -1
        else:
            raise Exception('get_profit2: prediction is neither 1 nor 0.')


def get_profit_betting_on_fav(x):
    min_val = min(x['home_odds'],  x['draw_odds'],  x['away_odds'])
    if x['actual'] == 'H':
        if x['home_odds'] == min_val:
            return x['home_odds']-1
        else:
            return -1
    elif x['actual'] == 'D':
        if x['draw_odds'] == min_val:
            return x['draw_odds']-1
        else:
            return -1
    elif x['actual'] == 'A':
        if x['away_odds'] == min_val:
            return x['away_odds']-1
        else:
            return -1
    else:
        raise Exception('full_time_result is not H, D or A.')


def get_feature_data(min_training_data_date='2013-08-01'):
    df = run_query(query="""select t1.*, m_h.manager home_manager,
     m_h.start_date home_manager_start, 
     m_a.manager away_manager, m_a.start_date away_manager_start 
     from main_fixtures t1 
     left join managers m_h 
     on t1.home_id = m_h.team_id 
     and (t1.date between m_h.start_date and date(m_h.end_date, '+1 day') 
     or t1.date > m_h.start_date and m_h.end_date is NULL) 
     left join managers m_a 
     on t1.away_id = m_a.team_id 
     and (t1.date between m_a.start_date and date(m_a.end_date, '+1 day') 
     or t1.date > m_a.start_date and m_a.end_date is NULL) 
     where t1.date > '{}'""".format(min_training_data_date))
    df=get_manager_features(df)
    df2 = run_query(query="select * from team_fixtures where date > '{}'".format(
        min_training_data_date))
    df2['date'] = pd.to_datetime(df2['date'])
    df2 = pd.merge(
        df2,
        df[['date', 'season', 'fixture_id', 'home_manager_age', 'away_manager_age',
            'home_manager_new', 'away_manager_new']],
        on=['date', 'season', 'fixture_id'],
        how="left")
    return df2


def get_team_model_performance(x, model_id, home_team=True):
    """Get the performance of the model at predicting a certain team"""
    team_id = x['home_id'] if home_team else x['away_id']
    date = x['date']
    season = x['season']
    perf = run_query(query="""select avg(correct) from historic_predictions where 
        model_id = '{}' and season = '{}' and home_id = {} 
        or away_id = {} and date < '{}'""".format(
            model_id, season, team_id, team_id, date))
    return perf.iloc[0, 0]


def apply_profit_weight(x):
    if x['target'] == 'H':
        return x['home_odds'] - 1
    elif x['target'] == 'D':
        return x['draw_odds'] - 1
    elif x['target'] == 'A':
        return x['away_odds'] - 1
    else:
        raise Exception('Error in apply_profit_weight()')


# ToDo: Player features
# Missing key players (top 3)
