import requests
import pandas as pd
import json
import dash
from flask import Flask
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table

requests.post("http://127.0.0.1:12345/update")
response = requests.get("http://127.0.0.1:12345/next_games")
data = response.json()
fixture_list = pd.DataFrame()

for i in range(10):
    fixture_list = fixture_list.append(pd.DataFrame({
        'kickoff_time': data.get('kickoff_time').get(str(i)),
        'home_team': data.get('home_name').get(str(i)),
        'home_id': data.get('home_id').get(str(i)),
        'away_team': data.get('away_name').get(str(i)),
        'away_id': data.get('away_id').get(str(i)),
     }, index=[i]))

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
    response = {k:float(v) for (k,v) in response.items()}
    response_df = pd.DataFrame(response, index=[i])
    predictions = predictions.append(response_df)
    print(response)

# Combine the latest fixtures and the predictions DataFrames()
df_cols = ['kickoff_time', 'home_team', 'away_team', 'H', 'D', 'A']
latest_preds = pd.concat([fixture_list, predictions], axis=1)[df_cols]
# Add odds for convenience
latest_preds['H_odds'] = round(1/latest_preds['H'], 2)
latest_preds['D_odds'] = round(1/latest_preds['D'], 2)
latest_preds['A_odds'] = round(1/latest_preds['A'], 2)

# ToDo: Get the latest bookmaker odds

# ToDo: Get the models performance on past data

# Create the dashboard
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    children=[
        dbc.Row([
            dbc.Col(
                [
                html.H2("Upcoming Fixtures"),
                dash_table.DataTable(
                    id='next_fixtures',
                    columns=[{"name": i, "id": i} for i in fixture_list.columns],
                    data=fixture_list.to_dict('records'),
                )]
            )]
        )
    ]
)

app.run_server(debug=False, port=8050)
