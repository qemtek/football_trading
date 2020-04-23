import os
import logging

from football_trading.src.utils.configuration import get_attribute

PROJECTSPATH = get_attribute('PROJECTSPATH')
IN_PRODUCTION = get_attribute('IN_PRODUCTION')
data_dir = os.path.join(PROJECTSPATH, 'football_trading', 'data')
plots_dir = os.path.join(PROJECTSPATH, 'football_trading', 'plots')
model_dir = os.path.join(PROJECTSPATH, 'football_trading', 'models')
training_data_dir = os.path.join(PROJECTSPATH, 'football_trading', 'data', 'training_data')
sql_dir = os.path.join(PROJECTSPATH, 'football_trading', 'sql')
tmp_dir = os.path.join(PROJECTSPATH, 'football_trading', 'tmp')
db_dir = get_attribute('DB_DIR')
RECREATE_DB = get_attribute('RECREATE_DB')

BFEX_USER = get_attribute('BFEX_USER')
BFEX_PASSWORD = get_attribute('BFEX_PASSWORD')
BFEX_APP_KEY = get_attribute('BFEX_APP_KEY')
BFEX_CERTS_PATH = get_attribute('BFEX_CERTS_PATH')

# Models covered in the project so far
available_models = ['XGBClassifier', 'XGBRegressor', 'LogisticRegression']
PRODUCTION_MODEL_NAME = get_attribute('PRODUCTION_MODEL_NAME')

# Logging configuration
FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(funcName)s:%(lineno)d - %(message)s")
LOG_FILE = f"{PROJECTSPATH}/football_trading/logs/model.log"


