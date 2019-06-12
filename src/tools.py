import sqlite3

# Connect to database
def connect_to_db(dir = None):

    # If no name is supplied, use the default name
    if dir == None: sqlite_file = '/Users/chriscollins/Documents/GitHub/football_trading/data/db.sqlite'

    # Establish a connection to the database
    conn = sqlite3.connect(sqlite_file)

    # Return the connection object
    return conn, conn.cursor()
