from src.utils.base_model import  get_logger
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
    X = test_dataset.drop('full_time_result_A')
    y = test_dataset['full_time_result_a']
    model.optimise_hyperparams(X=X, y=y)
    # ToDo: Pick a dataset where optimization improves thee model,
    #  then ensure it does this in the test

def test_train_model():
    pass