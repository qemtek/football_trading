import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import GridSearchCV, KFold

from football_trading.settings import PROJECTSPATH, training_data_dir
from football_trading.src.utils.base_model import time_function
from football_trading.src.utils.general import safe_open
from football_trading.src.models.templates.base_model import BaseModel
from football_trading.src.utils.logging import get_logger

logger = get_logger()


class SKLearnModel(BaseModel):
    """Anything that can be used by any SKLearn model goes in this class"""

    def __init__(self, test_mode=False, model_object=None, save_trained_model=True, load_trained_model=False,
                 load_model_date=None, compare_models=False, problem_name=None) -> None:
        super().__init__(model_object=model_object, load_trained_model=load_trained_model,
                         save_trained_model=save_trained_model, load_model_date=load_model_date,
                         compare_models=compare_models, test_mode=test_mode, problem_name=problem_name)
        self.param_grid = None
        # The metric used to evaluate model performance
        self.scoring = 'accuracy'

    @time_function(logger=logger)
    def optimise_hyperparams(self, *, X, y, param_grid=None) -> None:
        """Hyper-parameter optimisation function using GridSearchCV. Works for any SKLearn models"""

        logger.info("Optimising hyper-parameters. Param grid: {}".format(self.param_grid))
        param_grid = self.param_grid if param_grid is None else param_grid
        model = self.model_object() if self.params is None else self.model_object(**self.params)
        clf = GridSearchCV(model, param_grid, verbose=1, scoring=self.scoring, n_jobs=1)
        clf.fit(X[self.model_features], np.ravel(y))
        self.params = clf.best_params_
        logger.info('Best hyper-parameters: {}'.format(self.params))

    @time_function(logger=logger)
    def train_model(self, *, X, y, sample_weight=None, n_splits=10) -> None:
        """Train a model on 90% of the data and predict 10% using KFold validation,
        such that a prediction is made for all data"""

        logger.info("Training model.")
        kf = KFold(n_splits=n_splits)
        y = np.ravel(np.array(y))
        labels = list(np.sort(np.unique(y)))
        model_predictions = pd.DataFrame()
        sample_weight = np.ones(len(X)) if sample_weight is None else sample_weight
        for train_index, test_index in kf.split(X):
            model = self.model_object(**self.params).fit(
                X=X.iloc[train_index, :][self.model_features],
                y=y[train_index],
                sample_weight=sample_weight[train_index])
            preds = model.predict(X.iloc[test_index, :][self.model_features])
            preds_proba = model.predict_proba(X.iloc[test_index, :][self.model_features])
            actuals = y[test_index]
            model_predictions = model_predictions.append(
                pd.concat([
                    X.iloc[test_index, :],
                    pd.DataFrame(preds, columns=['pred'], index=X.iloc[test_index, :].index),
                    pd.DataFrame(preds_proba, columns=['predict_proba_' + str(label) for label in labels],
                                 index=X.iloc[test_index, :].index),
                    pd.DataFrame(actuals, columns=['actual'], index=X.iloc[test_index, :].index)], axis=1))
            # Add on a column to indicate whether the prediction was correct or not
            model_predictions['correct'] = model_predictions.apply(
                lambda x: 1 if x['pred'] == x['actual'] else 0, axis=1)
            # Save model predictions to the class
            self.model_predictions = model_predictions
            # Save training data used to train/evaluate the model
            self.training_data = {"X_train": X.iloc[train_index, :], "y_train": y[train_index],
                                  "X_test": X.iloc[test_index, :], "y_test": y[test_index]}
            # Assess the model performance using the first performance metric
            main_performance_metric = self.performance_metrics[0].__name__
            performance = self.performance_metrics[0](actuals, preds)
            # If the model performs better than the previous model, save it
            # ToDo: Returning 0 when there is no performance score only works
            #  for performance scores where higher is better
            if performance > self.performance.get(main_performance_metric, 0):
                self.trained_model = model
                for metric in self.performance_metrics:
                    metric_name = metric.__name__
                    self.performance[metric_name] = metric(actuals, preds)
            logger.info('Training finished. {}: {}'.format(
                str(main_performance_metric),
                str(performance)))
        # Save the data used to train the model
        data_save_dir = f"{training_data_dir}/{self.model_id}.joblib"
        with safe_open(data_save_dir, 'wb') as f_out:
            joblib.dump(self.training_data, f_out)
