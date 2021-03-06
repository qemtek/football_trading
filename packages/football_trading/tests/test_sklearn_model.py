import os

from sklearn.linear_model import LogisticRegression

from football_trading.src.utils.logging import get_logger
from football_trading.src.models.templates.sklearn_model import SKLearnModel
from football_trading.settings import PROJECTSPATH


logger = get_logger()


def test_optimise_hyperparameters(test_dataset):
    model = SKLearnModel(model_object=LogisticRegression, problem_name='test')
    # Initial model parameters (without tuning)
    model.params = {"max_iter": 100000}
    # Define a grid for hyper-parameter tuning
    model.param_grid = {
        "penalty": ["l2", "none"],
        "solver": ["newton-cg", "lbfgs", "sag", "saga"],
    }
    model.model_features = ['avg_goals_for_home', 'avg_goals_for_away']
    X = test_dataset.drop('full_time_result_H', axis=1)
    y = test_dataset['full_time_result_H']
    model.optimise_hyperparams(X=X, y=y)
    # Clean up files
    hp_dir = f"{PROJECTSPATH}/data/hyperparams/test.joblib"
    if os.path.exists(hp_dir):
        os.remove(hp_dir)
    # ToDo: Pick a dataset where optimization improves the model,
    #  then ensure it does this in the test
