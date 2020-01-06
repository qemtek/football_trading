import sqlite3
import pandas as pd

# Connect to database
def connect_to_db(dir = None):
    # If no name is supplied, use the default name
    if dir == None: sqlite_file = '/Users/chriscollins/Documents/GitHub/football_trading/data/db.sqlite'
    # Establish a connection to the database
    conn = sqlite3.connect(sqlite_file)
    # Return the connection object
    return conn, conn.cursor()


# Function to run a query on the DB while still keeping the column names. Returns a DataFrame
def run_query(cursor, query, params=[], return_data=True):
    # Run query
    cursor.execute(query, params)
    # Get column names and apply to the data frame
    if return_data:
        names = cursor.description
        name_list = []
        for name in names:
            name_list.append(name[0])
        # Convert the result into a DataFrame and add column names
        df = pd.DataFrame(cursor.fetchall(), columns=name_list)
        return df

