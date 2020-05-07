import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import datetime as dt

from src.utils.db import run_query, connect_to_db
from src.utils.team_id_functions import fetch_id, fetch_name


def hasNumbers(inputString):
    """Test whether the string contains numbers"""
    return bool(re.search(r"\d", inputString))


def fix_team_name(name):
    """Removes A.F.C. and F.C. from team names and deletes trailing whitespace"""
    name = name.replace("A.F.C.", "")
    name = name.replace("F.C.", "")
    name = name.strip()
    name = name.encode("UTF-8").decode()
    return name


def get_manager_data():
    """Download html content from wikipedia, break down with BeautifulSoup
    """
    # Connect to database
    conn = connect_to_db()
    url = "https://en.wikipedia.org/wiki/List_of_Premier_League_managers"
    website_url = requests.get(url).text
    soup = BeautifulSoup(website_url, features="html.parser")
    My_table = soup.find("table", {"class": "wikitable sortable plainrowheaders"})
    # All data can be found in 'a' and 'span' tags
    links = My_table.findAll("a")
    links2 = My_table.findAll("span")
    # Load data from links1
    link1_data = []
    for name in links:
        link1_data.append(name.get("title"))
    # Load data from links2
    link2_data = []
    for name in links2:
        link2_data.append(name.get("data-sort-value"))
    # Remove Nulls
    link2_data = [i for i in link2_data if i]
    link1_data = [i for i in link1_data if i]
    # Test2 manager name indexes
    name_indexes = []
    for i in range(0, len(link2_data)):
        if not hasNumbers(link2_data[i]):
            name_indexes.append(i)
    # Test3 manager name indexes
    manager = []
    country = []
    team = []
    for i in range(0, len(link1_data)):
        if i % 3 == 0:
            manager.append(link1_data[i])
        if i % 3 == 1:
            country.append(link1_data[i])
        if i % 3 == 2:
            team.append(link1_data[i])
    # Create a DataFrame to store all of the scraped data
    managers = pd.DataFrame()
    for i in range(0, len(name_indexes)):
        manager_index = name_indexes[i]
        try:
            next_manager = name_indexes[i + 1]
        except IndexError:
            next_manager = len(name_indexes) + 1
        num_elements = next_manager - manager_index
        manager_name = link2_data[manager_index]
        manager_sd = link2_data[manager_index + 1]

        # Remove the first 0 zeros from manager_sd
        manager_sd = manager_sd[8:-5]
        managers.loc[i, "manager"] = " ".join(str.split(manager[i])[0:2])
        managers.loc[i, "country"] = country[i]
        managers.loc[i, "team"] = fix_team_name(team[i])
        managers.loc[i, "from"] = manager_sd
        # If the manager has left, there will be a second row we must capture
        if num_elements == 3:
            manager_ed = link2_data[manager_index + 2]
            # Remove the first 0 zeros from manager_ed
            manager_ed = manager_ed[8:-5]
            managers.loc[i, "until"] = manager_ed
    # Replace club names with those used in the rest of the project
    # (e.g. Use Manchester City instead of Man City). Also get the team_id
    managers["team_id"] = managers["team"].apply(lambda x: fetch_id(x))
    managers["team"] = managers["team_id"].apply(lambda x: fetch_name(x))
    # Rename columns
    managers = managers[["manager", "team", "team_id", "from", "until"]]
    managers['from'] = pd.to_datetime(managers['from'])
    managers['until'] = pd.to_datetime(managers['until'])
    # If the until date of the last manager and from date of the next manager are the same,
    # # subtract 1 day from until of the previous manager
    managers['next_from'] = managers.groupby('team')['from'].shift(-1)
    managers['until'] = managers.apply(lambda x: x['until']
    if x['until'] != x['next_from'] else x['until'] - dt.timedelta(days=1), axis=1)
    df_dates = pd.DataFrame()
    for row in managers.iterrows():
        until = row[1]['until'] if \
            isinstance(row[1]['until'], pd._libs.tslibs.timestamps.Timestamp) \
            else dt.datetime.today()
        dates_between = pd.DataFrame([row[1]['from'] + dt.timedelta(days=x)
             for x in range((until - row[1]['from']).days + 1)], columns=['date'])
        dates_between['manager'] = row[1]['manager']
        dates_between['team'] = row[1]['team']
        dates_between['team_id'] = row[1]['team_id']
        df_dates = df_dates.append(dates_between)
    # Concatenate manager names when two managers have managed at once
    df_dates = df_dates.groupby(['date', 'team', 'team_id'])['manager'].apply(
        lambda x: ' & '.join(x)).reset_index()
    # Get the number of days between each row to identify missing data
    df_dates['date_lag'] = df_dates.groupby(['team', 'team_id'])['date'].apply(lambda x: x.shift(1))
    df_dates['date_diff'] = df_dates.apply(lambda x: (x['date'] - x['date_lag']).days, axis=1)
    # Create rows for missing data and add them to df_dates
    missing_data = df_dates[df_dates['date_diff'] > 1]
    missing_dates = pd.DataFrame()
    for row in missing_data.dropna().iterrows():
        dates_between = pd.DataFrame(
            [row[1]['date_lag'] + dt.timedelta(days=x+1)
             for x in range((row[1]['date'] - (row[1]['date_lag'] + dt.timedelta(days=1))).days)],
            columns=['date'])
        dates_between['manager'] = 'No Manager'
        dates_between['team'] = row[1]['team']
        dates_between['team_id'] = row[1]['team_id']
        missing_dates = missing_dates.append(dates_between)
    # Drop unnecessary columns and add the missing data rows
    df_dates.drop(['date_lag', 'date_diff'], axis=1, inplace=True)
    df_dates = df_dates.append(missing_dates).reset_index(drop=True).sort_values('date')
    # Create an indicator that tells us each time a team changes manager
    df_dates['last_manager'] = df_dates.groupby(['team', 'team_id'])['manager'].apply(lambda x: x.shift(1))
    df_dates['manager_change'] = df_dates.apply(
        lambda x: 1 if x['manager'] != x['last_manager'] else 0, axis=1)
    df_dates['manager_num'] = df_dates.groupby(['team', 'team_id'])['manager_change'].cumsum()
    # Aggregate the manager data to get a start/end date for each managerial spell
    min_dates = df_dates.groupby(
        ['team', 'team_id', 'manager', 'manager_num'])['date'].min().reset_index()
    max_dates = df_dates.groupby(
        ['team', 'team_id', 'manager', 'manager_num'])['date'].max().reset_index()
    # Add on managers who are still in power
    df_current = df_dates.groupby(
        ['team', 'team_id', 'manager', 'manager_num'])['date']
    manager_dates = pd.merge(min_dates, max_dates, on=['team', 'team_id', 'manager', 'manager_num']).reset_index(drop=True)
    manager_dates.columns = ['team', 'team_id', 'manager', 'manager_num', 'from', 'until']
    manager_dates = manager_dates.groupby(
        ['team', 'team_id', 'manager', 'manager_num', 'from'])['until'].max().reset_index()
    manager_dates = manager_dates.groupby(
        ['team', 'team_id', 'manager', 'manager_num', 'until'])['from'].min().reset_index()
    manager_dates = manager_dates.groupby(
        ['team', 'team_id', 'manager', 'manager_num', 'from'])['until'].max().reset_index()
    # Drop and recreate the table we are going to populate
    run_query(query="DROP TABLE IF EXISTS managers", return_data=False)
    run_query(query="CREATE TABLE managers (manager INT, team TEXT, "
                    "team_id INTEGER, start_date DATE, end_date DATE)",
              return_data=False)
    for row in manager_dates.iterrows():
        params = [
            str(row[1]["manager"]),
            str(row[1]["team"]),
            int(row[1]["team_id"]),
            str(row[1]["from"].date()),
            str(row[1]["until"].date()),
        ]
        run_query(query="INSERT INTO managers (manager, team, team_id, start_date, end_date) "
                        "VALUES(?, ?, ?, ?, ?)", params=params, return_data=False)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    get_manager_data()
