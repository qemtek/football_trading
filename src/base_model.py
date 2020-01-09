import pickle
import xgboost as xgb
import numpy as np
import os
import pandas as pd
import logging

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score


logger = logging.getLogger('XGBoostModel')

class Model:
    """Anything that can be used by all models goes in this class"""
    def __init__(self):
        pass

    def prerprocess(self, X):
        pass

    def train_model(self, X, y):
        pass

    def predict_proba(self, X):
        pass

    def predict_classes(self, X):
        pass


class XGBoostModel(Model, X, y):
    """Everything specific to the XGBoost goes in this class"""
    def __init__(self):
        self.param_grid = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}
        self.params = {'n_estimators': 100}
        self.balanced_accuracy = 0
        X = self.prerprocess(X)
        self.train_model()
        self.optimise_hyperparams()

    def train_model(self, X, y):
        """Train a model on 90% of the data and predict 10% using KFold validation,
        such that a prediction is made for all data"""
        kf = KFold(n_splits=10)
        model_predictions = pd.DataFrame()
        for train_index, test_index in kf.split(X):
            xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
            predictions = xgb_model.predict(X[test_index])
            actuals = y[test_index]
            model_predictions = model_predictions.append(
                pd.concat([X[test_index], predictions, actuals], axis=1))
            balanced_accuracy = balanced_accuracy_score(actuals, predictions)
            # If the model performs better than the previous model, save it
            if balanced_accuracy > self.balanced_accuracy:
                self.model = xgb_model
                self.balanced_accuracy = balanced_accuracy

    def optimise_hyperparams(self, X, y, param_grid=None):
        param_grid = self.param_grid if param_grid is None else param_grid
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        xgb_model = xgb.XGBClassifier()
        clf = GridSearchCV(xgb_model, param_grid, verbose=1)
        clf.fit(X_train, y_train)
        # Train a second model using the default parameters
        clf2 = xgb.XGBClassifier(params=self.params)
        clf2.fit(X_train, y_train)
        # Compare these params to existing params. If they are better, use them.
        clf_performance = balanced_accuracy_score(y_test, clf.predict(X_test))
        clf2_performance = balanced_accuracy_score(y_test, clf2.predict(X_test))
        if clf2_performance > clf_performance:
            print('')
            self.params = clf.params

        self.params = clf.best_params_

    @staticmethod
    def save_model(model):
        # The sklearn API models are picklable
        print("Pickling sklearn API models")
        # must open in binary format to pickle
        pickle.dump(model, open(os.path.join('data', "fpl_predict_model.pkl"), "wb"))

