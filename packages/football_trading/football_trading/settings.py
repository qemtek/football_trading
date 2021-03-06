import logging
import pathlib

import football_trading
from football_trading.src.utils.config import get_attribute

# Project specific credentials
#PROJECTSPATH = get_attribute('PROJECTSPATH')
PROJECTSPATH = pathlib.Path(football_trading.__file__).resolve().parent

# Logging configuration
FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(funcName)s:%(lineno)d - %(message)s")
LOG_FILE = f"{PROJECTSPATH}/logs/model.log"

IN_PRODUCTION = get_attribute('IN_PRODUCTION')
data_dir = f"{PROJECTSPATH}/data"
plots_dir = f"{PROJECTSPATH}/plots"
model_dir = f"{PROJECTSPATH}/models"
training_data_dir = f"{PROJECTSPATH}/data/training_data"
sql_dir = f"{PROJECTSPATH}/sql"
tmp_dir = f"{PROJECTSPATH}/tmp"
DB_DIR = get_attribute('DB_DIR')
RECREATE_DB = get_attribute('RECREATE_DB')
LOCAL=get_attribute('LOCAL')

# S3 credentials
S3_BUCKET_NAME = get_attribute('S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = get_attribute('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = get_attribute('AWS_SECRET_ACCESS_KEY')

# Betfair Exchange API credentials
BFEX_USER = get_attribute('BFEX_USER')
BFEX_PASSWORD = get_attribute('BFEX_PASSWORD')
BFEX_APP_KEY = get_attribute('BFEX_APP_KEY')
BFEX_CERTS_PATH = get_attribute('BFEX_CERTS_PATH')

# Models covered in the project so far
available_models = ['XGBClassifier', 'XGBRegressor', 'LogisticRegression']
PRODUCTION_MODEL_NAME = get_attribute('PRODUCTION_MODEL_NAME')
