import os

from src.utils.configuration import get_attribute

PROJECTSPATH = get_attribute('PROJECTSPATH')
data_dir = os.path.join(PROJECTSPATH, 'data')
plots_dir = os.path.join(PROJECTSPATH, 'plots')
model_dir = os.path.join(PROJECTSPATH, 'models')
db_dir = get_attribute('DB_DIR')
RECREATE_DB = get_attribute('RECREATE_DB')

BFEX_USER = get_attribute('BFEX_USER')
BFEX_PASSWORD = get_attribute('BFEX_PASSWORD')
BFEX_APP_KEY = get_attribute('BFEX_APP_KEY')
BFEX_CERTS_PATH = get_attribute('BFEX_CERTS_PATH')

# Credentials required to access the Betfair Exchange API.
betfair_credentials = {
    'betfairlightweight': {
        'username': BFEX_USER,
        'password': BFEX_PASSWORD,
        'app_key': BFEX_APP_KEY,
        'certs': BFEX_CERTS_PATH
    }
}

# Models covered in the project so far
available_models = ['XGBClassifier', 'XGBRegressor', 'LogisticRegression']
