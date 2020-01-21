import logging
import pandas as pd
import os
from src.utils.db import connect_to_db, run_query
from configuration import available_models, model_dir
import joblib
import datetime as dt
from src.utils.base_model import load_model
import shap
import numpy as np

logger = logging.getLogger('XGBoostModel')

class Model:
    """Anything that can be used by all models goes in this class"""
    def __init__(self):
        self.params = None
        self.model = None
        self.model_type = None
        self.training_data_query = \
            """select t1.*, m_h.manager home_manager, m_h.start_date home_manager_start, 
            m_a.manager away_manager, m_a.start_date away_manager_start 
            from main_fixtures t1 
            left join managers m_h 
            on t1.home_id = m_h.team_id 
            and (t1.date between m_h.start_date and date(m_h.end_date, '+1 day') 
            or t1.date > m_h.start_date and m_h.end_date is NULL) 
            left join managers m_a 
            on t1.away_id = m_a.team_id 
            and (t1.date between m_a.start_date and date(m_a.end_date, '+1 day') 
            or t1.date > m_a.start_date and m_a.end_date is NULL) 
            where t1.date > '2013-08-01'"""

    def get_data(self, X):
        """Get model features, given a DataFrame of match info"""
        pass

    def train_model(self, X, y):
        pass

    def predict(self, X):
        X = self.preprocess(X)
        return self.model.predict_proba(X) if self.model is not None else None

    def preprocess(self, X):
        """Apply preprocessing steps to data"""
        return np.array(X)

    def get_training_data(self):
        conn, cursor = connect_to_db()
        # Get all fixtures after game week 8, excluding the last game week
        df = run_query(cursor, self.training_data_query)
        return df

    def save_model(self):
        if self.model is None:
            logger.error("Trying to save a model that is None, aborting.")
        else:
            file_name = self.model_type + '_' + str(dt.datetime.today().date()) + '.joblib'
            save_dir = os.path.join(model_dir, file_name)
            logger.info("Saving model to {} with joblib.".format(save_dir))
            joblib.dump(self.model, open(save_dir, "wb"))

    def load_model(self, model_type, date=None):
        """Wrapper for the load model function in utils"""
        model = load_model(model_type, date=date)
        if model is None:
            return False
        else:
            # Set the attributes of the model to those of the class
            self.model = model
            self.params = model.get_params()
            return True

    @staticmethod
    def get_categorical_features(X):
        """Get a list of categorical features in the data"""
        categoricals = []
        for col, col_type in X.dtypes.iteritems():
            if col_type == 'O':
                categoricals.append(col)
        return categoricals

    @staticmethod
    def fill_na_values(self, X):
        """Fill NA values in the data"""
        # ToDo: Add a data structure that specifies how to fill NA's for every column
        pass

    @staticmethod
    def get_shap_explainer(model, X, plot_force=False):
        """Explain model predictions using SHAP"""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # visualize the first prediction's explanation
        # (use matplotlib=True to avoid Javascript)
        if plot_force:
            shap.force_plot(explainer.expected_value[0], shap_values[0])
        return explainer, shap_values
