"""This script contains all the functions used to generate data for the dashboard"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from football_trading.src.utils.logging import get_logger
from football_trading.settings import training_data_dir, IN_PRODUCTION, model_dir

logger = get_logger(logger_name='dashboard')


def load_production_model():
    production_model_dir = os.path.join(model_dir, 'in_production')
    production_model = os.listdir(production_model_dir)
    production_model = [m for m in production_model if '.DS' not in m]
    if len(production_model) > 0:
        logger.warning('There are two models in the in_production folder.. Picking the first one.')
    production_model_dir = os.path.join(production_model_dir, production_model[0])
    logger.info(f'Loading production model from {production_model_dir}')
    with open(production_model_dir, 'rb') as f_in:
        production_model = joblib.load(f_in)
    # Get the ID of the production model
    production_model_id = production_model.model_id
    return production_model, production_model_id, production_model_dir


def get_team_home_away_performance(historic_df, current_season):
    """Get the performance of the model for each team, at home and away
    """
    team_df = historic_df[historic_df['season'] == current_season]
    home_team_perf = team_df.groupby('home_team')['correct'].mean().round(2)
    away_team_perf = team_df.groupby('away_team')['correct'].mean().round(2)
    combined_team_perf = pd.concat([home_team_perf, away_team_perf], axis=1).reset_index()
    combined_team_perf.columns = ['Team Name', 'Accuracy when Home', 'Accuracy when Away']
    return combined_team_perf


def get_performance_by_season(historic_df_all, production_model_id):
    """Get the performance of all models, grouped by season
    """
    # If we're in production, just look at the production model
    if IN_PRODUCTION:
        historic_df_all = historic_df_all[historic_df_all['model_id'] == production_model_id]
    time_perf = pd.DataFrame(historic_df_all.groupby(['season', 'model_id'])['correct'].mean().round(2)).reset_index()
    date_perf = pd.DataFrame(historic_df_all.groupby(['season', 'model_id'])['date'].max()).reset_index()
    time_perf = pd.merge(date_perf, time_perf, on=['season', 'model_id'])
    time_perf.columns = ['Season', 'Model ID', 'Date', 'Accuracy']
    return time_perf


def get_cumulative_profit_view(historic_df_all, all_model_ids, production_model_id):
    """Get the cumulative profit gained form each model over time"""

    # If we're in production, just look at the production model
    if IN_PRODUCTION:
        historic_df_all = historic_df_all[historic_df_all['model_id'] == production_model_id]
    profit_perf = pd.DataFrame()
    for model in all_model_ids:
        df = pd.DataFrame(historic_df_all[historic_df_all['model_id'] == model])
        df = df.sort_values('date').reset_index()
        profit_perf = profit_perf.append(
            df.groupby(['date', 'model_id'])['profit'].sum().cumsum().reset_index())
    profit_perf.columns = ['Date', 'Model ID', 'Profit']
    return profit_perf


def get_cumulative_profit_from_bof(historic_df):
    """Get the cumulative profit from betting on the favourite
    """
    profit_perf_bof = pd.DataFrame(historic_df.sort_values('date').groupby(
        ['date'])['profit_bof'].sum().cumsum()).reset_index()
    profit_perf_bof.columns = ['Date', 'Profit']
    return profit_perf_bof


def get_form_dif_view(historic_df_all, production_model_id):
    """Get the performance of each model, grouped by the difference in form between teams
    """
    # If we're in production, just look at the production model
    if IN_PRODUCTION:
        historic_df_all = historic_df_all[historic_df_all['model_id'] == production_model_id]
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


def get_historic_features(predictions, model_names):
    historic_df_all = pd.DataFrame()
    for model in model_names:
        historic_df_all = historic_df_all.append(predictions[predictions['model_id'] == model.split('.')[0]])
    # Sort rows by date
    historic_df_all['date'] = pd.to_datetime(historic_df_all['date'])
    historic_df_all = historic_df_all.sort_values('date')
    # Add on a column stating whether the model was correct or not
    historic_df_all['correct'] = historic_df_all.apply(
        lambda x: 1 if x['pred'] == x['actual'] else 0, axis=1)
    # Add on a rounded fixture ID (grouping fixtures into sets of 10)
    historic_df_all['rounded_fixture_id'] = np.ceil(historic_df_all['fixture_id'] / 10)
    # Add on the year (to group data by year)
    historic_df_all['year'] = historic_df_all['date'].apply(lambda x: pd.to_datetime(x).year)
    # Add on the week number (to group data by week)
    historic_df_all['week_num'] = historic_df_all.apply(
        lambda x: str(x['year']) + str(round(x['rounded_fixture_id'])).zfill(2), axis=1)
    # Get the IDs of all models in the data
    all_model_ids = historic_df_all['model_id'].unique()
    # Get the training data corresponding to all models
    historic_training_data = pd.DataFrame()
    for id in all_model_ids:
        td = joblib.load(f"{training_data_dir}/{id}.joblib").get('train_test_data')
        df = pd.concat([td.get('X_train'), td.get('X_test')], axis=0).reset_index(drop=True)
        targets = pd.DataFrame(pd.concat(
            [pd.Series(td.get('y_train')), pd.Series(td.get('y_test'))],
            axis=0).reset_index(drop=True), columns=['full_time_result'])
        df = pd.concat([df, targets], axis=1)
        try:
            historic_training_data = historic_training_data.append(df)
        except TypeError:
            historic_training_data = historic_training_data.append(df.get('X_train'))
    historic_df_all = pd.merge(
        historic_df_all, historic_training_data,
        on=['home_team', 'away_team', 'date', 'season', 'fixture_id'])
    return historic_df_all, all_model_ids
