import datetime as dt
import os
import joblib
import pytest

from sklearn.linear_model import LogisticRegression

from packages.football_trading.src.models.templates.base_model import BaseModel
from packages.football_trading import model_dir


@pytest.mark.basemodel
@pytest.mark.timeout(20)
def test_save_model():
    # Save a dummy model
    model = BaseModel(model_object=LogisticRegression, problem_name='test')
    model.trained_model = LogisticRegression()
    model.trained_model.model_id = model.model_id
    model.save_model()
    # Try loading the dummy model and check that everything is correct
    file_name = '{}_{}_{}.joblib'.format(
        model.problem_name, model.model_type, str(dt.datetime.today().date()))
    save_dir = os.path.join(model_dir, file_name)
    loaded_model = joblib.load(open(save_dir, "rb"))
    # Clean up file
    os.remove(save_dir)
    assert model.trained_model is loaded_model.trained_model


@pytest.mark.basemodel
@pytest.mark.timeout(20)
def test_load_model():
    # Save a dummy model
    model = BaseModel(model_object=LogisticRegression, problem_name='test')
    model.trained_model = LogisticRegression()
    model.trained_model.model_id = model.model_id
    model.save_model()
    model2 = BaseModel(
        model_object=LogisticRegression,
        problem_name='test',
        load_trained_model=True)
    # Clean up files
    file_name = '{}_{}_{}.joblib'.format(
        model.problem_name, model.model_type, str(dt.datetime.today().date()))
    save_dir = os.path.join(model_dir, file_name)
    # Clean up file
    os.remove(save_dir)
    assert model is model2


def test_get_logger():
    # Test that the logger object is returned
    # Test that the logger creates the appropriate files
    # Test that the logger prints INFO and ERROR messages to the correct log files
    pass


