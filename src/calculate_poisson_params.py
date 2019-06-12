import pandas as pd
import sqlite3
from src.tools import connect_to_db

# Calculates the mean and sd of the poisson distribution of goals for/against for each team.

# Connect to database
conn, cursor = connect_to_db()

# Extract data
df = cursor.execute('select team_name, date, season, goals_for, goals_against from team_fixtures')

# Get column names and apply to the data frame
names = cursor.description
name_list = []
for name in names:
    name_list.append(name[0])

df = pd.DataFrame(cursor.fetchall(), columns=name_list).reset_index(drop=True)

#df.groupby()
#df.rolling(window=2).mean()