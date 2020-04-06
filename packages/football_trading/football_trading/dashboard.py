# ToDo: Model performance after an unexpected loss
# ToDo: Run t-tests to find out whether the accuracy on any of the weeks is significantly different
# ToDo: Cluster the matchups, look for similar groups. Look at the accuracy of each cluster
# ToDo: Add Game week, see if it improves the model accuracy on certain weeks
# ToDo: Add squad value as a feature (probably from wikipedia)
# ToDo: Add the latest bookmaker odds to latest_preds
# ToDo: Load training data and join if onto historic_df_all
# ToDo: Get a view of historic predictions where its 1 row per team.. Then
#  look at the profit from betting on each team

import pandas as pd
import dash
from flask import Flask
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import plotly.graph_objects as go
import os
import joblib

from football_trading.settings import model_dir, training_data_dir
from football_trading.src.utils.logging import get_logger
from football_trading.src.utils.dashboard import (
    get_form_dif_view, get_team_home_away_performance, get_performance_by_season,
    get_cumulative_profit_view, get_cumulative_profit_from_bof)
from football_trading.src.utils.db import run_query

logger = get_logger(logger_name='dashboard')


def get_dashboard_app(server=None):

    # Get the latest predictions
    latest_preds = run_query(query='select * from latest_predictions')
    # Get the names of models saved in the models directory
    model_names = os.listdir(model_dir)
    predictions = run_query(query='select * from historic_predictions')
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
    historic_df_all['rounded_fixture_id'] = np.ceil(historic_df_all['fixture_id']/10)
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
        df = joblib.load(os.path.join(training_data_dir, id + '.joblib'))
        try:
            historic_training_data = historic_training_data.append(df)
        except TypeError:
            historic_training_data = historic_training_data.append(df.get('X_train'))
    historic_df_all = pd.merge(
        historic_df_all, historic_training_data,
        on=['home_team', 'away_team', 'date', 'season', 'fixture_id'])

    # Load the in-production model
    production_model_dir = os.path.join(model_dir, 'in_production')
    production_model = os.listdir(production_model_dir)
    if len(production_model) > 0:
        logger.warning('There are two models in the in_production folder.. Picking the first one.')
    production_model_dir = os.path.join(production_model_dir, production_model[0])
    logger.info(f'Loading production model from {production_model_dir}')
    with open(production_model_dir, 'rb') as f_in:
        production_model = joblib.load(f_in)
    # Get the ID of the production model
    production_model_id = production_model.model_id
    # Get the current season
    current_season = run_query(query='select max(season) from main_fixtures').iloc[0, 0]
    # Get the historic predictions for the production model
    historic_df = historic_df_all[historic_df_all['model_id'] == production_model_id]
    # Get the performance of the model for each team, at home and away
    combined_team_perf = get_team_home_away_performance(historic_df=historic_df, current_season=current_season)
    # Get the performance of all models, grouped by season
    time_perf = get_performance_by_season(historic_df_all=historic_df_all)
    # Get the cumulative profit gained form each model over time
    profit_perf = get_cumulative_profit_view(historic_df_all=historic_df_all, all_model_ids=all_model_ids)
    # Get the cumulative profit from betting on the favourite
    profit_perf_bof = get_cumulative_profit_from_bof(historic_df=historic_df)
    # Get the performance of each model, grouped by the difference in form between teams
    form_diff_acc = get_form_dif_view(historic_df_all=historic_df_all)

    if server is not None:
        logger.info(f'Server passed of type {str(type(server))}. It will only be used if its of type Flask.')
    # Create the dashboard
    app = dash.Dash(__name__,
                    # Use the server if its a Flask object, else create our own
                    server=server if isinstance(server, Flask) else True,
                    external_stylesheets=[dbc.themes.BOOTSTRAP],
                    url_base_pathname='/dashboard/')
    app.config.suppress_callback_exceptions = True

    if len(latest_preds) > 0:
        upcoming_fixture_table = dash_table.DataTable(
            id='next_fixtures',
            columns=[{"name": i, "id": i} for i in latest_preds.columns],
            data=latest_preds.to_dict('records'))
    else:
        upcoming_fixture_table = html.P('No upcoming fixtures available. Was there a pandemic recently?')

    app.layout = html.Div(
        children=[
            dbc.Row([
                dbc.Col([
                    html.H2("Upcoming Fixtures"),
                    upcoming_fixture_table,
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Model Performance Plots"),
                    html.P("The following sections show the performance of the model, split into different views. "
                           "These views allow us to see where the model is underperforming, so that we can "
                           "design new features to help mitigate its shortcomings.")
                ])
            ]),
            dbc.Row([
                dbc.Col([

                    html.H4("Profit Over Time (Cumulative Sum)"),
                    html.P("This section shows the cumulative profit by date of the different "
                           "models. We use 10-fold cross-validation to train the model, saving the "
                           "predictions in each fold. Because of this, we have predictions for 100% "
                           "of the data which is very useful for analysing model performance"
                           "when you don't have much data."),
                    dcc.Graph(
                            id='profit_by_date',
                            figure={
                                'data': [
                                    go.Scatter(
                                        x=profit_perf[profit_perf['Model ID'] == model_id]['Date'],
                                        y=profit_perf[profit_perf['Model ID'] == model_id]['Profit'],
                                        mode="lines+markers",
                                        name='_'.join(model_id.split('_')[:-1]),
                                    ) for model_id in all_model_ids] +
                                    [go.Scatter(
                                        x=profit_perf_bof['Date'],
                                        y=profit_perf_bof['Profit'],
                                        mode="lines+markers",
                                        name='Betting on Favourite'
                                    )],
                                'layout':
                                    go.Layout(
                                        title="Profit Over Time (Cumulative Sum)",
                                        clickmode='event+select',
                                        height=600,
                                    )
                            })],
                    width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Model Accuracy for PL Teams - Home and Away"),
                    html.P("Below you can see the accuracy of the model at predicting "
                           "for each team, split by whether the team was at home or away. "
                           "Future strategies may involve only betting on teams that the "
                           "model has a good accuracy with. The accuracy of the model is "
                           "mainly down to the consistency of results for the team."),
                    dcc.Graph(
                        id='accuracy_home_away',
                        figure={
                            'data': [
                                {'x': combined_team_perf['Team Name'],
                                 'y': combined_team_perf['Accuracy when Home'],
                                 'type': 'bar', 'name': 'Home'},
                                {'x': combined_team_perf['Team Name'],
                                 'y': combined_team_perf['Accuracy when Away'],
                                 'type': 'bar', 'name': 'Away'}
                            ],
                            'layout': {
                                'title': 'Model Accuracy for PL Teams - Home and Away'
                            }
                        }
                    )
                ], width=12)
            ]),
            dbc.Row(
                dbc.Col([
                    html.H4("Model Accuracy Over Time"),
                    html.P("Below you can see the accuracy of each model over time. Notice how "
                           "the model under-performs in certain seasons. The outlier is the year "
                           "Leicester won the league."),
                    dcc.Graph(
                            id='accuracy_over_time',
                            figure={
                                'data': [
                                    go.Scatter(
                                        x=time_perf[time_perf['Model ID'] ==
                                                    model_id].sort_values('Date')['Date'],
                                        y=time_perf[time_perf['Model ID'] ==
                                                    model_id].sort_values('Date')['Accuracy'],
                                        mode="lines+markers",
                                        name='_'.join(model_id.split('_')[:-1])
                                    ) for model_id in all_model_ids],
                                'layout': go.Layout(
                                        title="Model Accuracy Over Time",
                                        clickmode='event+select',
                                        height=600,
                                    )
                            })],
                    width=12)
            ),
            dbc.Row(
                dbc.Col([
                    html.H4("Accuracy vs Form Difference"),
                    html.P("This figure shows the performance of the model, split by the "
                           "average goal difference of the opponents in the last 8 games. "
                           "The more difficult matchups are when the teams have similar performance. "
                           "We can also see the effect of the home advantage here."),
                    dcc.Graph(
                            id='form_diff_acc',
                            figure={
                                'data': [
                                    go.Scatter(
                                        x=form_diff_acc[form_diff_acc['Model ID'] ==
                                                    model_id].sort_values('Form Dif')['Form Dif'],
                                        y=form_diff_acc[form_diff_acc['Model ID'] ==
                                                    model_id].sort_values('Form Dif')['Correct'],
                                        mode="lines+markers",
                                        name='_'.join(model_id.split('_')[:-1])
                                    ) for model_id in all_model_ids],
                                'layout': go.Layout(
                                        title="Accuracy vs Form Difference (home_goal_dif_last_7 - "
                                              "away_goal_dif_last_7)",
                                        clickmode='event+select',
                                        height=600,
                                    )
                            })],
                    width=12)
            )
        ]
    )
    # Return the api
    return app


if __name__ == '__main__':
    app = get_dashboard_app()
    app.run_server(port=8050)
