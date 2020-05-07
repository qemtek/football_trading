import sqlite3
import pandas as pd
import os
import time

from src.utils.logging import get_logger
from settings import DB_DIR, S3_BUCKET_NAME
from src.utils.s3_tools import download_from_s3, list_files

logger = get_logger()


def connect_to_db(path_to_db=None):
    """Connect to local sqlite3 database
    """
    # If no name is supplied, use the default name
    sqlite_file = DB_DIR if path_to_db is None else path_to_db
    # Establish a connection to the database
    if os.path.exists(sqlite_file):
        conn = sqlite3.connect(sqlite_file)
    else:
        raise Exception(f"DB does not exist at {sqlite_file}")
    # Return the connection object
    return conn


def run_query(*, query, params=None, return_data=True, path_to_db=None) -> pd.DataFrame:
    """Function to run a query on the DB while still keeping the column names. Returns a DataFrame
    """
    if os.path.exists(query):
        with open(query, 'r') as f_in:
            query = ''
            for line in f_in.readlines():
                query = query + line
    with connect_to_db(path_to_db) as conn:
        cursor = conn.cursor()
        split_query = query.split(';')
        if len(split_query) > 1:
            for query in split_query:
                # Run query
                print('Multiple queries detected, not returning data')
                cursor.execute(query, params if params is not None else [])
        else:
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


def get_db(local):
    if local:
        if not os.path.exists(f"{DB_DIR}"):
            raise Exception('LOCAL is True and the DB does not exist locally')
    else:
        local_exists = os.path.exists(f"{DB_DIR}")
        file_data = list_files(prefix='', bucket='football-trading')
        files = [f for f in file_data if f.get('Key') == 'db.sqlite']
        assert len(files) <= 1, 'More than 1 file named db.sqlite in S3'
        remote_exists = len(files) > 0
        if local_exists and not remote_exists:
            latest = 'local'
            pass
        if remote_exists and not local_exists:
            latest = 'remote'
        if not local_exists and not remote_exists:
            raise Exception('Cant find the DB in the local or remote locations')
        if local_exists and remote_exists:
            last_modified_remote = pd.to_datetime(files[0].get('LastModified'), utc=True)
            last_modified_local = os.path.getmtime(f"{DB_DIR}")
            last_modified_local = pd.to_datetime(time.ctime(last_modified_local), utc=True)
            if last_modified_remote > last_modified_local:
                latest = 'remote'
            elif last_modified_local > last_modified_remote:
                latest = 'local'
            elif last_modified_local == last_modified_remote:
                latest = 'both'
        if latest == 'remote':
            download_from_s3(local_path=f"{DB_DIR}", s3_path='db.sqlite', bucket=S3_BUCKET_NAME)
