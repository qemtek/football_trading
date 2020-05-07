import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from settings import PROJECTSPATH, S3_BUCKET_NAME
from src.utils.logging import get_logger
from src.utils.general import safe_open, time_function
from src.models.templates.base_model import BaseModel
from src.utils.s3_tools import upload_to_s3

logger = get_logger()


class SKLearnModel(BaseModel):
    """Anything that can be used by any SKLearn model goes in this class"""

    def __init__(self, test_mode=False, model_object=None, save_trained_model=True, load_trained_model=False,
                 load_model_date=None, compare_models=False, problem_name=None, is_classifier=False,
                 local=True) -> None:
        super().__init__(model_object=model_object, load_trained_model=load_trained_model,
                         save_trained_model=save_trained_model, load_model_date=load_model_date,
                         compare_models=compare_models, test_mode=test_mode, problem_name=problem_name,
                         is_classifier=is_classifier)
        self.is_classifier = is_classifier
        self.param_grid = None
        self.local = local
        # The metric used to evaluate model performance
        self.scoring = 'roc_auc' if is_classifier else 'neg_mean_absolute_error'

    @time_function(logger=logger)
    def optimise_hyperparams(self, *, X, y, param_grid=None) -> None:
        """Hyper-parameter optimisation function using GridSearchCV. Works for any SKLearn models
        """
        logger.info("Optimising hyper-parameters. Param grid: {}".format(self.param_grid))
        param_grid = self.param_grid if param_grid is None else param_grid
        model = self.model_object() if self.params is None else self.model_object(**self.params)
        clf = GridSearchCV(model, param_grid, verbose=1, scoring=self.scoring, n_jobs=10, cv=3)
        clf.fit(X[self.model_features], np.ravel(y))
        self.params = clf.best_params_
        logger.info('Best hyper-parameters: {}'.format(self.params))
        # Save hyper-params
        save_dir = f"{PROJECTSPATH}/data/hyperparams/{self.problem_name}.joblib"
        with safe_open(save_dir, 'wb') as f_out:
            joblib.dump(clf.best_params_, f_out)
            logger.info(f'Hyper-params saved to {save_dir}')

    @time_function(logger=logger)
    def train_model(self, *, X, y, sample_weight=None, n_splits=10) -> None:
        """Train a model using KFold validation, such that a prediction is made for all data
        """
        logger.info("Training model.")
        kf = StratifiedKFold(n_splits=n_splits)
        y = np.ravel(np.array(y))
        labels = list(np.sort(np.unique(y)))
        model_predictions = pd.DataFrame()
        sample_weight = np.ones(len(X)) if sample_weight is None else sample_weight
        for train_index, test_index in kf.split(X, y):
            model = self.model_object(**self.params, n_jobs=3).fit(
                X=X.iloc[train_index, :][self.model_features],
                y=y[train_index],
                sample_weight=sample_weight[train_index])
            # ToDo: Add fold number so we can check out certain folds that
            #  have exceptionally high/low performance
            preds = model.predict(X.iloc[test_index, :][self.model_features])
            actuals = y[test_index]
            predictions = pd.concat([
                X.iloc[test_index, :],
                pd.DataFrame(preds, columns=['pred'], index=X.iloc[test_index, :].index),
                pd.DataFrame(actuals, columns=['actual'], index=X.iloc[test_index, :].index)], axis=1)
            if self.is_classifier:
                preds_proba = pd.Series(model.predict_proba(X.iloc[test_index, :][self.model_features])[:, 1])
                predictions = pd.concat([predictions.reset_index(drop=True), preds_proba], axis=1)
            model_predictions = model_predictions.append(predictions)
            # Add on a column to indicate whether the prediction was correct or not
            if self.is_classifier:
                model_predictions['correct'] = model_predictions.apply(
                    lambda x: 1 if x['pred'] == x['actual'] else 0, axis=1)
            # Assess the model performance using the first performance metric
            main_performance_metric = self.performance_metrics[0].__name__
            performance = self.performance_metrics[0](actuals, preds)
            # If the model performs better than the previous model, save it
            if performance > self.performance.get(main_performance_metric, 0):
                self.trained_model = model
                self.trained_model.model_features = self.model_features
                for metric in self.performance_metrics:
                    metric_name = metric.__name__
                    self.performance[metric_name] = metric(actuals, preds)
                # Save training data used to train/evaluate the model
                self.training_data = {"X_train": X.iloc[train_index, :], "y_train": y[train_index],
                                      "X_test": X.iloc[test_index, :], "y_test": y[test_index]}
            logger.info('Training finished. {}: {}'.format(str(main_performance_metric), str(performance)))
        # Save model predictions to the class
        self.model_predictions = model_predictions
