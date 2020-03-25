import os

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, 'data')
plots_dir = os.path.join(project_dir, 'plots')
model_dir = os.path.join(data_dir, 'models')
db_dir = os.path.join(data_dir, 'db.sqlite')

# List of available models
available_models = ['XGBClassifier', 'LGBM', 'LogisticRegression']
