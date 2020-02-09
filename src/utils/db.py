import sqlite3
import pandas as pd
from configuration import db_dir


def connect_to_db(path_to_db=None):
    """# Connect to local sqlite3 database"""
    # If no name is supplied, use the default name
    sqlite_file = db_dir if path_to_db is None else path_to_db
    # Establish a connection to the database
    conn = sqlite3.connect(sqlite_file)
    # Return the connection object
    # ToDo: Just return the conn
    return conn


def run_query(query, params=None, return_data=True):
    """Function to run a query on the DB while still keeping the
    column names. Returns a DataFrame"""
    with connect_to_db() as conn:
        cursor = conn.cursor()
        # Run query
        cursor.execute(query, params if params is not None else [])
        # Get column names and apply to the data frame
        if return_data:
            names = cursor.description
            name_list = []
            for name in names:
                name_list.append(name[0])
            # Convert the result into a DataFrame and add column names
            df = pd.DataFrame(cursor.fetchall(), columns=name_list)
            return df
