import logging
import joblib
import os
import time
from functools import wraps
import pandas as pd

from configuration import available_models, model_dir, project_dir
from src.utils.general import safe_open


# Check if the model type is an accepted one
def load_model(model_type, date=None, logger=None, keyword=None):
    # Set up logging
    if logger is None:
        logger = logging.getLogger('load_model')
    # Check that the requested model type is one that we have
    if model_type not in available_models:
        logger.error("load_model: model_type must be one of {}. "
                     "Returning None".format(', '.join(available_models)))
        return None
    # Check if the supplied date is in the correct format
    model_date = date if date is not None else 'latest'
    logger.info("Attempting to load {} model for date: "
                "{}".format(model_type, model_date))
    # Load all model directories
    models = os.listdir(model_dir)
    # Filter for the model type
    models_filtered = [model for model in models if model.find(model_type) != -1]
    # Filter for keyword
    if keyword is not None:
        models_filtered = [model for model in models if model.split('_')[0] == keyword]
    # Filter for the date (if requested)
    if date is not None:
        models_filtered = [model for model in models if model.find(date) != -1]
    # Order models by date and pick the latest one
    if len(models_filtered) > 0:
        models_filtered.sort(reverse=True)
        model_name = models_filtered[0]
        logger.info("load_model: Loading model with filename: {}".format(model_name))
        with safe_open(os.path.join(model_dir, model_name), "rb") as f_in:
            return joblib.load(f_in)
    else:
        return None


def get_logger(log_name='model'):
    log_dir = os.path.join(project_dir, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # Create logger
    logger = logging.getLogger(log_name)
    # Setup a logger to handle DEBUG messages
    l1 = logging.FileHandler(os.path.join(log_dir, '{}_debug.log'.format(log_name)))
    l1.setLevel(logging.DEBUG)
    # Setup a logger to handle INFO messages
    l2 = logging.FileHandler(os.path.join(log_dir, '{}.log'.format(log_name)))
    l2.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    l1.setFormatter(formatter)
    l2.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(l1)
    logger.addHandler(l2)
    return logger


def time_function(logger):
    """A decorator function to time how long a function takes to run"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            logger.debug(
                "Function {} completed in {} seconds".format(
                    func.__name__, round(time.time() - start_time)
                )
            )
            return result
        return wrapper
    return decorator


def suspend_logging(logger):
    """A decorator that suspends logging for a function and all functions
    called by that function"""
    # ToDo: Understand how this works
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            previousloglevel = logger.getEffectiveLevel()
            try:
                return func(*args, **kwargs)
            finally:
                logger.setLevel(previousloglevel)
        return wrapper
    return decorator


def fill_na_values(X):
    """Fill NA values in the data"""
    # ToDo: Add a data structure that specifies how to fill NA's for every column
    pass
