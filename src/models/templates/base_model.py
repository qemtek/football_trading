import os
import datetime as dt
import joblib

from sklearn.metrics import balanced_accuracy_score, accuracy_score

from configuration import model_dir
from src.utils.base_model import load_model, time_function, get_logger
from src.utils.db import run_query
from src.utils.general import safe_open

logger = get_logger()


class BaseModel:
    """Anything that can be used by any model goes in this class"""
    def __init__(self,
                 model_object=None,
                 load_trained_model=True,
                 save_trained_model=True,
                 test_mode=False,
                 load_model_date=None,
                 problem_name=None,
                 compare_models=True):
        # Store all arguments passed to __init__ inside the class,
        # so we know what they were later
        self.model_object = model_object
        self.load_trained_model = load_trained_model
        self.save_trained_model = save_trained_model
        self.test_mode = test_mode
        self.load_model_date = load_model_date
        self.problem_name = problem_name
        self.compare_models = compare_models
        # The date this class was instantiated
        self.creation_date = str(dt.datetime.today().date())
        # The name of the model you want to use
        self.model_type = self.model_object.__name__
        # A unique identifier for this model
        self.model_id = "{}_{}_{}_{}".format(
            self.problem_name, self.model_type, self.creation_date, str(abs(hash(dt.datetime.today()))))
        # Store the trained model here
        self.trained_model = self.load_model(load_model_date) if load_trained_model else None
        # Load previous best model
        self.previous_model = self.load_model()
        # The name of all features in the model, specified as a list
        self.model_features = None
        # Model parameters
        self.params = None
        # A place to store predictions made by the model
        self.model_predictions = None
        # What name to give the problem the model is trying to solve
        self.problem_name = problem_name
        # A list of performance metrics (pass the functions, they must
        # take actuals, predictions as the first and second arguments
        self.performance_metrics = [balanced_accuracy_score, accuracy_score]
        # A dictionary to store the performance metrics for the trained model
        self.performance = {}
        # Name of the target variable (or variables, stored in a list)
        self.target = None
        # A query used to retrieve training data
        self.training_data_query = None

    @time_function(logger=logger)
    def get_training_data(self):
        df = run_query(self.training_data_query)
        return df

    @time_function(logger=logger)
    def preprocess(self, X):
        return X

    @time_function(logger=logger)
    def optimise_hyperparams(self, X, y, param_grid=None):
        return NotImplemented

    @time_function(logger=logger)
    def train_model(self, X, y):
        return NotImplemented

    @time_function(logger=logger)
    def predict(self, X):
        """Predict on a new set of data"""
        X = self.preprocess(X[self.model_features])
        return self.trained_model.predict(X[self.model_features])

    @time_function(logger=logger)
    def save_model(self):
        """Save a trained model to the models directory"""
        if self.trained_model is None:
            logger.error("Trying to save a model that is None, aborting.")
        else:
            # Save the model ID inside the model object (so we know which
            # model made which predictions in the DB)
            self.trained_model.model_id = self.model_id
            self.trained_model.model_features = self.model_features
            self.trained_model.performance_metrics = self.performance_metrics
            self.trained_model.performance = self.performance
            file_name = self.model_id + '.joblib'
            save_dir = os.path.join(model_dir, file_name)
            logger.info("Saving model to {} with joblib.".format(save_dir))
            with safe_open(save_dir, "wb") as f_out:
                joblib.dump(self.trained_model, f_out)

    @time_function(logger=logger)
    def load_model(self, date=None):
        """Wrapper for the load model function in utils"""
        model = load_model(self.model_type, date=date, keyword=self.problem_name)
        if model is None:
            logger.warning('No available models of type {}.'.format(self.model_type))
            return None
        else:
            # Set the attributes of the model to those of the class
            self.model_id = model.model_id
            if hasattr(model, 'params'):
                self.params = model.params
            if hasattr(model, 'model_features'):
                self.model_features = model.model_features
            if hasattr(model, 'performance_metrics'):
                self.performance_metrics = model.performance_metrics
            if hasattr(model, 'performance'):
                self.performance = model.performance
            else:
                logger.warning('The loaded model has no get_params method, '
                               'cannot load model parameters.')
            return model

    @time_function(logger=logger)
    def compare_latest_model(self):
        main_performance_metric = self.performance_metrics[0].__name__
        new_performance = self.performance.get(main_performance_metric)
        old_performance = self.previous_model.performance.get(main_performance_metric)
        if new_performance > old_performance:
            logger.info('New model beats previous model. Replacing this model')
            logger.info('{}: Previous Model: {}, New Model: {}'.format(
                main_performance_metric, old_performance, new_performance
            ))
            return True
        else:
            logger.info('New model does not beat previous model.')
            logger.info('{}: Previous Model: {}, New Model: {}'.format(
                main_performance_metric, old_performance, new_performance
            ))
            return False
