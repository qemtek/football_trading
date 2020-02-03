import os

# Change this to your own directory
project_dir = '/Users/chriscollins/Documents/GitHub/football_trading'

data_dir = os.path.join(project_dir, 'data')
plots_dir = os.path.join(project_dir, 'plots')
model_dir = os.path.join(data_dir, 'models')
db_dir = os.path.join(data_dir, 'db.sqlite')

# List of available models
available_models = ['XGBClassifier', 'LGBM']
