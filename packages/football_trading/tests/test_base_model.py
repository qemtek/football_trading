import os
import joblib
import pytest

from sklearn.linear_model import LogisticRegression

from football_trading.src.models.templates.base_model import BaseModel
from football_trading.settings import model_dir


@pytest.mark.basemodel
@pytest.mark.timeout(20)
def test_save_model():
    # Save a dummy model
    model = BaseModel(model_object=LogisticRegression, problem_name='test')
    model.trained_model = LogisticRegression()
    model.trained_model.model_id = model.model_id
    model.save_model()
    # Try loading the dummy model and check that everything is correct
    file_name = f'{model.model_id}.joblib'
    save_dir = os.path.join(model_dir, file_name)
    loaded_model = joblib.load(open(save_dir, "rb"))
    # Clean up file
    os.remove(save_dir)
    assert model.model_id == loaded_model.model_id, 'Save model has not worked correctly'


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
    file_name = f'{model.model_id}.joblib'
    save_dir = os.path.join(model_dir, file_name)
    # Clean up file
    os.remove(save_dir)
    assert model.model_id == model2.model_id, 'Load model has not worked correctly'
