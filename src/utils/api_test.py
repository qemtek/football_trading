import requests
import pandas as pd
import json

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

input = fixture_list.loc[0, ['kickoff_time', 'home_id', 'away_id']]
input2 = {
    "date": str(input['kickoff_time']),
    "home_id": str(input['home_id']),
    "away_id": str(input['away_id']),
    "season": "19/20",
}
response = requests.post("http://127.0.0.1:12345/predict", json=input2)
print(response.json())