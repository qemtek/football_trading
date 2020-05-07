from src.utils.logging import get_logger
from src.models.templates.sklearn_model import SKLearnModel
from sklearn.linear_model import LogisticRegression


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
    # ToDo: Pick a dataset where optimization improves the model,
    #  then ensure it does this in the test


def test_train_model():
    # ToDo: Complete
    pass
