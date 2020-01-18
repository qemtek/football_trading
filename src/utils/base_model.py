import logging
import joblib
import os
from configuration import available_models, model_dir


# Check if the model type is an accepted one
def load_model(model_type, date=None, logger=None):
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
    # Filter for the date (if requested)
    if date is not None:
        models_filtered = [model for model in models if model.find(date) != -1]
    # Order models by date and pick the latest one
    try:
        models_filtered.sort(reverse=True)
        model_name = models_filtered[0]
        logger.info("load_model: Loading model with filename: {}".format(model_name))
        model = joblib.load(open(os.path.join(model_dir, model_name), "rb"))
        return model
    except FileNotFoundError:
        logger.error("Chosen model could not be loaded, returning None")
        return None
    except TypeError:
        logger.error("No models could be found, returning None")
        return None
