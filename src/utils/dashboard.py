"""This script contains all the functions used to generate data for the dashboard"""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def get_team_home_away_performance(historic_df, current_season):
    """Get the performance of the model for each team, at home and away"""

    team_df = historic_df[historic_df['season'] == current_season]
    home_team_perf = team_df.groupby('home_team')['correct'].mean().round(2)
    away_team_perf = team_df.groupby('away_team')['correct'].mean().round(2)
    combined_team_perf = pd.concat([home_team_perf, away_team_perf], axis=1).reset_index()
    combined_team_perf.columns = ['Team Name', 'Accuracy when Home', 'Accuracy when Away']
    return combined_team_perf


def get_performance_by_season(historic_df_all):
    """Get the performance of all models, grouped by season"""

    time_perf = pd.DataFrame(historic_df_all.groupby(['season', 'model_id'])['correct'].mean().round(2)).reset_index()
    date_perf = pd.DataFrame(historic_df_all.groupby(['season', 'model_id'])['date'].max()).reset_index()
    time_perf = pd.merge(date_perf, time_perf, on=['season', 'model_id'])
    time_perf.columns = ['Season', 'Model ID', 'Date', 'Accuracy']
    return time_perf


def get_cumulative_profit_view(historic_df_all, all_model_ids):
    """Get the cumulative profit gained form each model over time"""

    profit_perf = pd.DataFrame()
    for model in all_model_ids:
        df = pd.DataFrame(historic_df_all[historic_df_all['model_id'] == model])
        df = df.sort_values('date').reset_index()
        profit_perf = profit_perf.append(
            df.groupby(['date', 'model_id'])['profit'].sum().cumsum().reset_index())
    profit_perf.columns = ['Date', 'Model ID', 'Profit']
    return profit_perf


def get_cumulative_profit_from_bof(historic_df):
    """Get the cumulative profit from betting on the favourite"""

    profit_perf_bof = pd.DataFrame(historic_df.sort_values('date').groupby(
        ['date'])['profit_bof'].sum().cumsum()).reset_index()
    profit_perf_bof.columns = ['Date', 'Profit']
    return profit_perf_bof


def get_form_dif_view(historic_df_all):
    """Get the performance of each model, grouped by the difference in form between teams"""

    historic_df_all['home_form'] = historic_df_all['avg_goals_for_home'] - historic_df_all['avg_goals_against_home']
    historic_df_all['away_form'] = historic_df_all['avg_goals_for_away'] - historic_df_all['avg_goals_against_away']
    historic_df_all['form_dif'] = historic_df_all['home_form'] - historic_df_all['away_form']
    historic_df_all.loc[historic_df_all['form_dif'] > 2, 'form_dif'] = 2
    historic_df_all.loc[historic_df_all['form_dif'] < -2, 'form_dif'] = -2
    historic_df_all['form_dif'] = round(historic_df_all['form_dif'] * 5) / 5
    form_dif_acc = historic_df_all.groupby(['form_dif', 'model_id'])['correct'].mean().reset_index()
    form_dif_acc.columns = ['Form Dif', 'Model ID', 'Correct']
    return form_dif_acc


def cluster_features(df):
    """Find clusters in the data using the features input into the model"""

    historic_df_features = df.drop(
        ['index', 'fixture_id', 'date', 'home_id', 'away_id',
        'home_team', 'away_team', 'season', 'year',
        'week_num', 'full_time_result', 'pred', 'model_id'],
        axis=1)
    cluster = DBSCAN()
    scaler = StandardScaler()
    historic_df_features = scaler.fit_transform(historic_df_features)
    res = cluster.fit(historic_df_features)
    return res
