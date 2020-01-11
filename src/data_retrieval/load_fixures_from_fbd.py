import pandas as pd
import numpy as np
from src.utils.team_id_functions import fetch_id
from datetime import datetime
from src.utils.db import connect_to_db

# Extract data from .csv files hosted at football-data.co.uk
def extract_data_from_fbd(url, table_name, connection_url=None):

    # Connect to database
    conn, cursor = connect_to_db(connection_url)

    # Pull the csv into a pandas data frame
    fixtureData = pd.read_csv(url, skipinitialspace=True,
                              error_bad_lines=False, keep_default_na=False)

    # Get the length of the data frame
    sLength = len(fixtureData['HomeTeam'])

    # Create an additional column for fixture_id, home_id and away_id. Full with random values for now
    fixtureData['fixture_id'] = pd.Series(np.random.randn(sLength), index=fixtureData.index)
    fixtureData['home_id'] = pd.Series(np.random.randn(sLength), index=fixtureData.index)
    fixtureData['away_id'] = pd.Series(np.random.randn(sLength), index=fixtureData.index)

    # Replace dodgy column names.
    fixtureData.columns = fixtureData.columns.str.replace('>', '_greater_than_').\
        str.replace('<', '_less_than_').str.replace('.', 'p')

    # Loop through each row of the data
    for i in range(0, len(fixtureData.index)):

        # Change the game date format to a more appropriate format
        try:
            try:
                date_corrected = datetime.strptime(fixtureData.Date[i], '%d/%m/%y').strftime('%Y-%m-%d')
            except:
                date_corrected = datetime.strptime(fixtureData.Date[i], '%d/%m/%Y').strftime('%Y-%m-%d')

            date_corrected = datetime.strptime(date_corrected, '%Y-%m-%d')

            # The next section determines which season the game is in, e.g. 16/17.

            # Extract the year and month
            year = date_corrected.year
            month = date_corrected.month

            # If we are in the second half of the season
            if month in [1, 2, 3, 4, 5, 6]:
                # The string is last_year / this_year
                season = str('{0:02d}'.format(int(str(year)[-2:]) - 1)) + '/' + str(year)[-2:]
            else:
                # The string is this_year / next_year
                season = str(year)[-2:] + '/' + str('{0:02d}'.format(int(str(year)[-2:]) + 1))

            # Get the over/under 2.5 odds (they are named differently in older .csv's)
            try:
                # Try the newer format
                o2p5 = fixtureData.BbMx_greater_than_2p5[i]
                u2p5 = fixtureData.BbMx_less_than_2p5[i]
            except:
                # Try the older format
                o2p5 = fixtureData.B365_greater_than_2p5[i]
                u2p5 = fixtureData.B365_less_than_2p5[i]

            # Get the Asian Handicap odds
            try:
                # Try the newer format
                ahHome = fixtureData.BbMxAHH[i]
                ahAway = fixtureData.BbMxAHA[i]
                ahHandicap = fixtureData.BbAHh[i]
            except:
                # Try the older format
                ahHome = fixtureData.B365AHH[i]
                ahAway = fixtureData.B365AHA[i]
                ahHandicap = fixtureData.B365AH[i]

            # Load all parameters into the main_fixtures table in SQLite.
            try:
                # Define the parameters
                params = [
                    i+1,  # Fixture ID
                    fixtureData.HomeTeam[i],  # Home team name
                    fetch_id(fixtureData.HomeTeam[i], cursor),  # Home team ID
                    fixtureData.AwayTeam[i],  # Away team name
                    fetch_id(fixtureData.AwayTeam[i], cursor),  # Away team ID
                    date_corrected,  # Fixture date
                    int(fixtureData.FTHG[i]),  # Home goals (full time)
                    int(fixtureData.FTAG[i]),  # Away goals (full time)
                    int(fixtureData.HS[i]),  # Home shots
                    int(fixtureData.AS[i]),  # Away shots
                    fixtureData.FTR[i],  # Full time result
                    float(fixtureData.B365H[i]),  # Bet 365 home odds
                    float(fixtureData.B365D[i]),  # Bet 365 draw odds
                    float(fixtureData.B365A[i]),  # Bet 365 away odds
                    fixtureData.Referee[i],  # Referee name
                    o2p5,  # Over 2.5 goal odds
                    u2p5,  # Under 2.5 goal odds
                    ahHome,  # Asian Handicap home odds
                    ahAway,  # Asian Handicap away odds
                    ahHandicap,  #  Asian Handicap
                    season,  # Season name (e.g. 16/17)
                    int(fixtureData.HY[i]),  # Home yellow cards
                    int(fixtureData.AY[i]),  # Away yellow cards
                    int(fixtureData.HR[i]),  # Home red cards
                    int(fixtureData.AR[i])]  # Away red cards

                # Load the parameters into the table
                query = """
                    INSERT INTO {tn} (
                        fixture_id, home_team, home_id, away_team, away_id, date, 
                        home_score, away_score, home_shots, away_shots, full_time_result, 
                        b365_home_odds, b365_draw_odds, b365_away_odds, referee, 
                        over_2p5_odds, under_2p5_odds, ah_home_odds, ah_away_odds, 
                        ah_home_handicap, season, home_yellow_cards, away_yellow_cards, 
                        home_red_cards, away_red_cards) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        ?, ?, ?, ?, ?, ?)"""
                cursor.execute(query.format(tn=table_name), params)
            except:
                # Some dates have no valid format, skipping for now.
                # ToDo: count the number of missing rows
                a = 1

            # Commit the changes
            conn.commit()

        except:
            # Debug block
            a=1


# Update fixtures from football-data.co.uk
def update_fixtures_from_fbd(table_name='main_fixtures'):

    # Connect to the database
    conn, cursor = connect_to_db()
    # Drop and recrease the table we are going to populate
    cursor.execute("DROP TABLE IF EXISTS {tn}".format(tn=table_name))
    cursor.execute("""CREATE TABLE {tn} (fixture_id INTEGER, home_team TEXT, 
    home_id INTEGER, away_team TEXT, away_id INTEGER, date DATE, home_score INTEGER, 
    away_score INTEGER, home_shots INTEGER, away_shots INTEGER, full_time_result TEXT, 
    b365_home_odds REAL, b365_draw_odds REAL, b365_away_odds REAL, referee TEXT, 
    over_2p5_odds REAL, under_2p5_odds REAL, ah_home_odds REAL, ah_away_odds REAL, 
    ah_home_handicap REAL, season TEXT, home_yellow_cards INTEGER, 
    away_yellow_cards INTEGER, home_red_cards INTEGER, 
    away_red_cards INTEGER)""".format(tn=table_name))
    conn.commit()
    # Create the list of urls to get data from
    urls = ['http://www.football-data.co.uk/mmz4281/0304/E0.csv',
            'http://www.football-data.co.uk/mmz4281/0405/E0.csv',
            'http://www.football-data.co.uk/mmz4281/0506/E0.csv',
            'http://www.football-data.co.uk/mmz4281/0607/E0.csv',
            'http://www.football-data.co.uk/mmz4281/0708/E0.csv',
            'http://www.football-data.co.uk/mmz4281/0809/E0.csv',
            'http://www.football-data.co.uk/mmz4281/0910/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1011/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1112/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1213/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1314/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1415/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1516/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1617/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1718/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1819/E0.csv',
            'http://www.football-data.co.uk/mmz4281/1920/E0.csv']

    for url in urls:
        extract_data_from_fbd(url, table_name)

    # Close the connection
    conn.close()


if __name__ == "__main__":
    update_fixtures_from_fbd()
