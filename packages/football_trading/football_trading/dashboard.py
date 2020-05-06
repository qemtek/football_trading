# ToDo: Model performance after an unexpected loss
# ToDo: Run t-tests to find out whether the accuracy on any of the weeks is significantly different
# ToDo: Cluster the matchups, look for similar groups. Look at the accuracy of each cluster
# ToDo: Add Game week, see if it improves the model accuracy on certain weeks
# ToDo: Add squad value as a feature (probably from wikipedia)
# ToDo: Add the latest bookmaker odds to latest_preds
# ToDo: Load training data and join if onto historic_df_all
# ToDo: Get a view of historic predictions where its 1 row per team.. Then
#  look at the profit from betting on each team

import dash
from flask import Flask
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
import os
import joblib

from dash.dependencies import Input, Output

from football_trading.settings import model_dir, tmp_dir, LOCAL, DB_DIR, S3_BUCKET_NAME
from football_trading.src.utils.logging import get_logger
from football_trading.src.utils.dashboard import (
    get_form_dif_view, get_team_home_away_performance, get_performance_by_season,
    get_cumulative_profit_view, get_cumulative_profit_from_bof, get_historic_features,
    load_production_model)
from football_trading.src.utils.db import run_query
from football_trading.src.utils.s3_tools import download_from_s3


active_graphs = ['profit-by-date', 'accuracy_home_away', 'accuracy_over_time', 'form_diff_acc']

logger = get_logger(logger_name='dashboard')

# ToDo: If we're in production, just look at the in-production model compared with betting on the favourite.

if not LOCAL:
    logger.info('Attempting to download DB from S3..')
    try:
        if not os.path.exists(f"{DB_DIR}"):
            download_from_s3(local_path=f"{DB_DIR}", s3_path='db.sqlite', bucket=S3_BUCKET_NAME)
    except Exception as e:
        raise Exception(f'DB cannot be found in S3, or the access credentials are incorrect. Error: {e}')


def get_dashboard_app(server=None):
    # Get the latest predictions
    latest_preds = run_query(query='select * from latest_predictions')
    # Get the names of models saved in the models directory
    model_names = os.listdir(model_dir)
    predictions = run_query(query='select * from historic_predictions')
    # Get historic features
    historic_df_all, all_model_ids = get_historic_features(predictions, model_names)
    # Drop Duplicates
    historic_df_all = historic_df_all.drop_duplicates()
    # Load the in-production model
    production_model, production_model_id, production_model_dir = load_production_model()
    # Get the current season
    current_season = run_query(query='select max(season) from main_fixtures').iloc[0, 0]
    # Get the historic predictions for the production model
    historic_df = historic_df_all[historic_df_all['model_id'] == production_model_id]
    # Save latest_preds and all_model_ids to the /tmp file. We will use these to
    # determine whether the dashboard has been updated
    joblib.dump(latest_preds, f"{tmp_dir}/latest_preds.joblib")
    joblib.dump(model_names, f"{tmp_dir}/model_names.joblib")
    # Get the performance of the model for each team, at home and away
    combined_team_perf = get_team_home_away_performance(historic_df=historic_df, current_season=current_season)
    # Get the performance of all models, grouped by season
    time_perf = get_performance_by_season(
        historic_df_all=historic_df_all, production_model_id=production_model_id)
    # Get the cumulative profit gained form each model over time
    profit_perf = get_cumulative_profit_view(
        historic_df_all=historic_df_all, all_model_ids=all_model_ids, production_model_id=production_model_id)
    # Get the cumulative profit from betting on the favourite
    profit_perf_bof = get_cumulative_profit_from_bof(historic_df=historic_df)
    # Get the performance of each model, grouped by the difference in form between teams
    form_diff_acc = get_form_dif_view(historic_df_all=historic_df_all, production_model_id=production_model_id)

    if server is not None:
        logger.info(f'Server passed of type {str(type(server))}. It will only be used if its of type Flask.')
    # Create the dashboard
    app = dash.Dash(__name__,
                    # Use the server if its a Flask object, else create our own
                    server=server if isinstance(server, Flask) else True,
                    external_stylesheets=[dbc.themes.BOOTSTRAP],
                    url_base_pathname='/dashboard/')
    app.config.suppress_callback_exceptions = True

    # Show the latest predictions (if any exist)
    if len(latest_preds) > 0:
        upcoming_fixture_table = dash_table.DataTable(
            id='next_fixtures',
            columns=[{"name": i, "id": i} for i in latest_preds.columns],
            data=latest_preds.to_dict('records'))
    else:
        upcoming_fixture_table = html.P('No upcoming fixtures available. Was there a pandemic recently?')

    # Define the app layout
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
            ),
            dbc.Row(
                dbc.Col([
                    dcc.Interval(
                        id='interval-component',
                        interval=1 * 1000 * 60,  # in milliseconds, run every minute
                        n_intervals=0
                    ),
                    dcc.Textarea(id='update')]
                )
            )
        ]
    )

    @app.callback(Output('update', 'value'), [Input('interval-component', 'n_intervals')])
    def check_for_updates(interval):
        # ToDo: Why isnt this updating properly
        # Get the names of models saved in the models directory
        model_names = os.listdir(model_dir)
        latest_preds = run_query(query='select * from latest_predictions')
        model_names_local = joblib.load(f"{tmp_dir}/model_names.joblib")
        latest_preds_local = joblib.load(f"{tmp_dir}/latest_preds.joblib")
        # If there are any updates, save a file for each active graph, which will tell the dashboard to update
        if not model_names.sort() == model_names_local.sort() or not latest_preds.equals(latest_preds_local):
            logger.info('Updates to the tables have been detected, updating graphs...')
            for graph in active_graphs:
                joblib.dump(1, f"{tmp_dir}/{graph}.joblib")
            # Save the new values
            joblib.dump(latest_preds, f"{tmp_dir}/latest_preds.joblib")
            joblib.dump(model_names, f"{tmp_dir}/model_names.joblib")

    @app.callback(Output('profit-by-date', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_pbd(interval):
        if os.path.exists(f"{tmp_dir}/profit-by-date.joblib"):
            logger.info('Updating profit_by_date graph.')
            # Get the names of models saved in the models directory
            model_names = os.listdir(model_dir)
            predictions = run_query(query='select * from historic_predictions')
            # Get historic features
            historic_df_all, all_model_ids = get_historic_features(predictions, model_names)
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
            # Get the historic predictions for the production model
            historic_df = historic_df_all[historic_df_all['model_id'] == production_model_id]
            # Get the cumulative profit gained form each model over time
            profit_perf = get_cumulative_profit_view(historic_df_all=historic_df_all, all_model_ids=all_model_ids)
            # Get the cumulative profit from betting on the favourite
            profit_perf_bof = get_cumulative_profit_from_bof(historic_df=historic_df)
            os.remove(f"{tmp_dir}/profit-by-date.joblib")
            return {
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
            }

    @app.callback(Output('accuracy_home_away', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_aha(interval):
        if os.path.exists(f"{tmp_dir}/accuracy_home_away.joblib"):
            logger.info('Updating accuracy_home_away graph.')
            # Get the names of models saved in the models directory
            model_names = os.listdir(model_dir)
            predictions = run_query(query='select * from historic_predictions')
            # Get historic features
            historic_df_all, all_model_ids = get_historic_features(predictions, model_names)
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
            # Get the production model features
            historic_df = historic_df_all[historic_df_all['model_id'] == production_model_id]
            # Get the performance of the model for each team, at home and away
            combined_team_perf = get_team_home_away_performance(historic_df=historic_df, current_season=current_season)
            os.remove(f"{tmp_dir}/accuracy_home_away.joblib")
            return {
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

    @app.callback(Output('accuracy_over_time', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_aot(interval):
        if os.path.exists(f"{tmp_dir}/accuracy_over_time.joblib"):
            logger.info('Updating accuracy_over_time graph.')
            # Get the names of models saved in the models directory
            model_names = os.listdir(model_dir)
            predictions = run_query(query='select * from historic_predictions')
            # Get historic features
            historic_df_all, all_model_ids = get_historic_features(predictions, model_names)
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
            # Get the performance of all models, grouped by season
            time_perf = get_performance_by_season(
                historic_df_all=historic_df_all, production_model_id=production_model_id)
            os.remove(f"{tmp_dir}/accuracy_over_time.joblib")
            return {
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
                }

    @app.callback(Output('form_dif_acc', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_fda(interval):
        if os.path.exists(f"{tmp_dir}/form_dif_acc.joblib"):
            logger.info('Updating form_dif_acc graph.')
            # Get the names of models saved in the models directory
            model_names = os.listdir(model_dir)
            predictions = run_query(query='select * from historic_predictions')
            # Get historic features
            historic_df_all, all_model_ids = get_historic_features(predictions, model_names)
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
            # Get the performance of each model, grouped by the difference in form between teams
            form_diff_acc = get_form_dif_view(
                historic_df_all=historic_df_all, production_model_id=production_model_id)
            os.remove(f"{tmp_dir}/form_dif_acc.joblib")
            return {
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
                }

    # Return the api
    return app


if __name__ == '__main__':
    app = get_dashboard_app()
    app.run_server(port=8050)
