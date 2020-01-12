import sqlite3
import pandas as pd
from configuration import db_dir


def connect_to_db(dir=None):
    """# Connect to local sqlite3 database"""
    # If no name is supplied, use the default name
    sqlite_file = db_dir if dir is None else dir
    # Establish a connection to the database
    conn = sqlite3.connect(sqlite_file)
    # Return the connection object
    return conn, conn.cursor()


def run_query(cursor, query, params=[], return_data=True):
    """Function to run a query on the DB while still keeping the
    column names. Returns a DataFrame"""
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

