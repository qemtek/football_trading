import logging
import os
from configuration import model_dir
import joblib
import datetime as dt
from src.utils.base_model import load_model
import shap
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

logger = logging.getLogger('XGBoostModel')

class Model:
    """Anything that can be used by all models goes in this class"""
    def __init__(self):
        self.params = None
        self.trained_model = None
        self.model_type = None
        self.param_grid = None
        self.model_object = None
        self.performance_metrics = [accuracy_score]
        self.scoring = 'accuracy'

    def get_data(self, X):
        """Get model features, given a DataFrame of match info"""
        pass

    def train_model(self, X, y):
        pass

    def optimise_hyperparams(self, X, y, param_grid=None):
        """Hyperparameter optimisation function using GridSearchCV. Works for any sklearn models"""
        logger.info("Optimising hyper-parameters")
        param_grid = self.param_grid if param_grid is None else param_grid
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = self.model_object()
        clf = GridSearchCV(model, param_grid, verbose=1, scoring=self.scoring, n_jobs=1)
        clf.fit(X_train, y_train)
        # Train a second model using the default parameters
        clf2 = self.model_object(params=self.params)
        clf2.fit(X_train, y_train)
        # Compare these params to existing params. If they are better, use them.
        # Use the first listed performance metric
        performance_metric = self.performance_metrics[0]
        # Get predictions for the first classifier
        clf_predictions = clf.best_estimator_.predict(X_test)
        clf_performance = performance_metric(y_test, clf_predictions)
        # Get predictions for the second classifierr
        clf2_predictions = clf2.predict(X_test)
        clf2_performance = performance_metric(y_test, clf2_predictions)
        # Compare performance
        if clf2_performance > clf_performance:
            logger.info("Hyper-parameter optimisation improves on previous model, "
                        "saving hyperparameters.")
            self.params = clf.best_params_

    def predict(self, X):
        X = self.preprocess(X)
        return self.trained_model.predict_proba(X) if self.trained_model is not None else None

    def preprocess(self, X):
        """Apply preprocessing steps to data"""
        return np.array(X)

    def get_training_data(self):
        pass

    def save_model(self):
        if self.trained_model is None:
            logger.error("Trying to save a model that is None, aborting.")
        else:
            file_name = self.model_type + '_' + str(dt.datetime.today().date()) + '.joblib'
            save_dir = os.path.join(model_dir, file_name)
            logger.info("Saving model to {} with joblib.".format(save_dir))
            joblib.dump(self.trained_model, open(save_dir, "wb"))

    def load_model(self, model_type, date=None):
        """Wrapper for the load model function in utils"""
        model = load_model(model_type, date=date)
        if model is None:
            return False
        else:
            # Set the attributes of the model to those of the class
            self.trained_model = model
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
