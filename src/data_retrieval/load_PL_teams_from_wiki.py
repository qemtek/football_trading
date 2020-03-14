import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from src.utils.db import run_query, connect_to_db
from src.utils.team_id_functions import fetch_alternative_name, fetch_alternative_name2


def get_teams_from_wiki():
    # Connect to database
    conn = connect_to_db()
    website_url = requests.get(
        'https://en.wikipedia.org/wiki/List_of_Premier_League_clubs').text
    soup = BeautifulSoup(website_url)
    My_table = soup.find('div', {'class': 'timeline-wrapper'})
    links = My_table.findAll('area')
    teams = []
    for link in links:
        team = link.get('alt').encode('UTF-8').decode()
        # Convert &26 to &
        team = team.replace('%26', '&')
        # Remove ' A.F.C' and ' F.C'
        team = team.replace('A.F.C', '')
        team = team.replace('F.C', '')
        # Remove any '.'s
        team = team.replace('.', '')
        # Remove white space from the start and end
        team = team.lstrip().rstrip()
        teams.append(team)
    df = pd.DataFrame()
    df['team_name'] = teams
    df = df.sort_values('team_name')
    df['team_id'] = np.arange(len(df))+1
    # Load the names into the database
    run_query('drop table if exists team_ids', return_data=False)
    run_query('create table team_ids (team_name TEXT, team_id INTEGER, '
              'alternate_name TEXT, alternate_name2 TEXT)',
              return_data=False)
    for row in df.iterrows():
        params = [row[1]['team_name'], row[1]['team_id']]
        params.append(fetch_alternative_name(row[1]['team_name']))
        params.append(fetch_alternative_name2(row[1]['team_name']))
        run_query('insert into team_ids(team_name, team_id, '
                  'alternate_name, alternate_name2) values(?, ?, ?, ?)',
                  params=params, return_data=False)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    get_teams_from_wiki()
