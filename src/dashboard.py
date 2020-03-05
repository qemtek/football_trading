import requests
import pandas as pd
import dash
from flask import Flask
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import logging
import plotly.graph_objects as go

from src.utils.base_model import get_logger

logger = get_logger(log_name='dashboard')

#requests.post("http://127.0.0.1:12345/update")
response = requests.get("http://127.0.0.1:12345/next_games")
data = response.json()
fixture_list = pd.DataFrame()

season = '19/20'

logger = logging.getLogger('API')

for i in range(10):
    fixture_list = fixture_list.append(pd.DataFrame({
        'kickoff_time': data.get('kickoff_time').get(str(i)),
        'home_team': data.get('home_name').get(str(i)),
        'home_id': data.get('home_id').get(str(i)),
        'away_team': data.get('away_name').get(str(i)),
        'away_id': data.get('away_id').get(str(i)),
     }, index=[i]))

fixture_list = fixture_list.drop_duplicates()

pl_teams = [list(fixture_list['home_id']) + list(fixture_list['away_id'])]
predictions = pd.DataFrame(columns=['H', 'D', 'A'])
for i in range(len(fixture_list)):
    print(i)
    input = fixture_list.loc[i, ['kickoff_time', 'home_id', 'away_id']]
    print(input)
    input_dict = {
        "date": str(input['kickoff_time']),
        "home_id": str(input['home_id']),
        "away_id": str(input['away_id']),
        "season": "19/20",
    }
    response = requests.post("http://127.0.0.1:12345/predict", json=input_dict).json()
    # Convert the probabilities back to floats
    response = {k: float(v) for (k, v) in response.items()}
    response_df = pd.DataFrame(response, index=[i])
    predictions = predictions.append(response_df)
    print(response)

# Combine the latest fixtures and the predictions DataFrames()
df_cols = ['kickoff_time', 'home_team', 'away_team', 'H', 'D', 'A']
latest_preds = pd.concat([fixture_list, predictions], axis=1)[df_cols]
# # Add odds for convenience
# latest_preds['H_odds'] = round(1/latest_preds['H'], 2)
# latest_preds['D_odds'] = round(1/latest_preds['D'], 2)
# latest_preds['A_odds'] = round(1/latest_preds['A'], 2)

# Get the models performance on past data
response = requests.get("http://127.0.0.1:12345/all_historic_predictions").json()
historic_df = pd.DataFrame()
for (k, v) in response.items():
    historic_df[k] = pd.Series(np.transpose(pd.DataFrame(v, index=[0]))[0])


def get_model_correct(x):
    return 1 if x['pred'] == x['actual'] else 0


def get_year(x):
    return pd.to_datetime(x).year


historic_df = historic_df.sort_values('date')
historic_df['correct'] = historic_df.apply(lambda x: get_model_correct(x), axis=1)
historic_df['rounded_fixture_id'] = np.ceil(historic_df['fixture_id']/10)
historic_df['year'] = historic_df['date'].apply(lambda x: get_year(x))

historic_df['week_num'] = historic_df.apply(
    lambda x: str(x['year']) + str(round(x['rounded_fixture_id'])).zfill(2), axis=1)

historic_df_all = historic_df
all_model_ids = historic_df_all['model_id'].unique()
response = requests.get("http://127.0.0.1:12345/latest_model_id").json()
model_id = response.get('model_id')
historic_df = historic_df[historic_df['model_id'] == model_id]

team_df = historic_df[historic_df['season'] == season]
home_team_perf = team_df.groupby('home_team')['correct'].mean().round(2)
away_team_perf = team_df.groupby('away_team')['correct'].mean().round(2)
combined_team_perf = pd.concat([home_team_perf, away_team_perf], axis=1).reset_index()
combined_team_perf.columns = ['Team Name', 'Accuracy when Home', 'Accuracy when Away']

home_season_team_perf = historic_df.groupby(['home_team', 'season'])['correct'].mean()
away_season_team_perf = historic_df.groupby(['away_team', 'season'])['correct'].mean()

time_perf = pd.DataFrame(historic_df_all.groupby(['season', 'model_id'])['correct'].mean().round(2)).reset_index()
date_perf = pd.DataFrame(historic_df_all.groupby(['season', 'model_id'])['date'].max()).reset_index()
time_perf = pd.merge(date_perf, time_perf, on=['season', 'model_id'])
time_perf.columns = ['Season', 'Model ID', 'Date', 'Accuracy']

profit_perf = pd.DataFrame(historic_df_all.sort_values('date').groupby(
    ['date', 'model_id'])['profit'].sum().cumsum()).reset_index()
profit_perf.columns = ['Date', 'Model ID', 'Profit']

profit_perf_bof = pd.DataFrame(historic_df.sort_values('date').groupby(
    ['date'])['profit_bof'].sum().cumsum()).reset_index()
profit_perf_bof.columns = ['Date', 'Profit']




# ToDo: Get a view of historic predictions where its 1 row per team.. Then
#  look at the profit from betting on each team
team_profit_perf = None

# ToDo: Model performance after an unexpected loss
# ToDo: Run t-tests to find out whether the accuracy on any of the weeks is significantly different
# ToDo: Cluster the matchups, look for similar groups. Look at the accuracy of each group
# ToDo: Add Game week, see if it improves the model accuracy on certain weeks
# ToDo: Add squad value as a feature (probably from wikipedia)
# ToDo: Add the latest bookmaker odds to latest_preds

historic_df_all['home_form'] = historic_df_all['avg_goals_for_home'] - historic_df_all['avg_goals_against_home']
historic_df_all['away_form'] = historic_df_all['avg_goals_for_away'] - historic_df_all['avg_goals_against_away']
historic_df_all['form_dif'] = historic_df_all['home_form'] - historic_df_all['away_form']
historic_df_all.loc[historic_df_all['form_dif'] > 2, 'form_dif'] = 2
historic_df_all.loc[historic_df_all['form_dif'] < -2, 'form_dif'] = -2
historic_df_all['form_dif'] = round(historic_df_all['form_dif']*5)/5
form_dif_acc = historic_df_all.groupby(['form_dif', 'model_id'])['correct'].mean().reset_index()
form_dif_acc.columns = ['Form Dif', 'Model ID', 'Correct']

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_features(df):
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


# Create the dashboard
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    children=[
        dbc.Row([
            dbc.Col([
                html.H2("Upcoming Fixtures"),
                dash_table.DataTable(
                    id='next_fixtures',
                    columns=[{"name": i, "id": i} for i in latest_preds.columns],
                    data=latest_preds.to_dict('records'),
                )]
            )]),
        dbc.Row([
            dbc.Col([
                html.H2("Model Performance Plots"),
                dcc.Graph(
                        id='profit_by_date',
                        figure={
                            'data': [
                                go.Scatter(
                                    x=profit_perf[profit_perf['Model ID'] == model_id]['Date'],
                                    y=profit_perf[profit_perf['Model ID'] == model_id]['Profit'],
                                    mode="lines+markers",
                                    name='_'.join(model_id.split('_')[-1]),
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
            dbc.Col(
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
                                    name='_'.join(model_id.split('_')[-1])
                                ) for model_id in all_model_ids],
                            'layout': go.Layout(
                                    title="Model Accuracy Over Time",
                                    clickmode='event+select',
                                    height=600,
                                )
                        }),
                width=12)
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                        id='form_diff_acc',
                        figure={
                            'data': [
                                go.Scatter(
                                    x=form_dif_acc[form_dif_acc['Model ID'] ==
                                                model_id].sort_values('Form Dif')['Form Dif'],
                                    y=form_dif_acc[form_dif_acc['Model ID'] ==
                                                model_id].sort_values('Form Dif')['Correct'],
                                    mode="lines+markers",
                                    name='_'.join(model_id.split('_')[-1])
                                ) for model_id in all_model_ids],
                            'layout': go.Layout(
                                    title="Accuracy vs Form Difference (home_gd - away_gd)",
                                    clickmode='event+select',
                                    height=600,
                                )
                        }),
                width=12)
        )
    ]
)


app.run_server(debug=False, port=8050)
