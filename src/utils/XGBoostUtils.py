import numpy as np
import pandas as pd


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
    team_name = row['home_team' if type == 'home' else 'away_team']
    fixture_id = row['fixture_id']
    season = row['season']
    # Filter for the team/season
    df_filtered = team_data[(team_data['team_name'] == team_name) &
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
    while (last_games.iloc[count] == 1):
        count += 1
    return count