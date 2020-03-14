import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from configuration import project_dir
from src.utils.base_model import time_function, get_logger
from src.models.templates.base_model import BaseModel

logger = get_logger()


class SKLearnModel(BaseModel):
    """Anything that can be used by any SKLearn model goes in this class"""
    def __init__(self,
                 test_mode=False,
                 model_object=None,
                 save_trained_model=True,
                 load_trained_model=False,
                 load_model_date=None,
                 problem_name=None):
        super().__init__(model_object=model_object,
                         load_trained_model=load_trained_model,
                         save_trained_model=save_trained_model,
                         load_model_date=load_model_date,
                         test_mode=test_mode,
                         problem_name=problem_name)
        self.param_grid = None
        # The metric used to evaluate model performance
        self.scoring = 'accuracy'

    @time_function(logger=logger)
    def optimise_hyperparams(self, X, y, param_grid=None):
        """Hyperparameter optimisation function using GridSearchCV. Works for any sklearn models"""
        logger.info("Optimising hyper-parameters")
        param_grid = self.param_grid if param_grid is None else param_grid
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = self.model_object() if self.params is None else self.model_object(**self.params)
        clf = GridSearchCV(model, param_grid, verbose=1, scoring=self.scoring, n_jobs=1)
        clf.fit(X_train, np.ravel(y_train))
        # Train a second model using the default parameters
        clf2 = self.model_object() if self.params is None else self.model_object(**self.params)
        clf2.fit(X_train, np.ravel(y_train))
        # Compare these params to existing params. If they are better, use them.
        # Use the first listed performance metric
        performance_metric = self.performance_metrics[0]
        # Get predictions for the first classifier
        clf_predictions = clf.best_estimator_.predict(X_test)
        clf_performance = performance_metric(y_test, clf_predictions)
        # Get predictions for the second classifier
        clf2_predictions = clf2.predict(X_test)
        clf2_performance = performance_metric(y_test, clf2_predictions)
        # Compare performance
        if clf2_performance > clf_performance:
            logger.info("Hyper-parameter optimisation improves on previous model, "
                        "saving hyperparameters.")
            self.params = clf.best_params_

    @time_function(logger=logger)
    def train_model(self, X, y, sample_weight=None):
        """Train a model on 90% of the data and predict 10% using KFold validation,
        such that a prediction is made for all data"""
        logger.info("Training model.")
        kf = KFold(n_splits=10)
        y = np.ravel(np.array(y))
        labels = list(np.sort(np.unique(y)))
        model_predictions = pd.DataFrame()
        for train_index, test_index in kf.split(X):
            model = self.model_object().fit(
                X=X.iloc[train_index, :][self.model_features],
                y=y[train_index], sample_weight=sample_weight[train_index])
            preds = model.predict(X.iloc[test_index, :][self.model_features])
            preds_proba = model.predict_proba(X.iloc[test_index, :][self.model_features])
            actuals = y[test_index]
            model_predictions = model_predictions.append(
                pd.concat([
                    X.iloc[test_index, :],
                    pd.DataFrame(preds, columns=['pred'], index=X.iloc[test_index, :].index),
                    pd.DataFrame(preds_proba,
                                 columns=['predict_proba_' + str(label) for label in labels],
                                 index=X.iloc[test_index, :].index),
                    pd.DataFrame(actuals, columns=['actual'], index=X.iloc[test_index, :].index)],
                    axis=1))
            # Add on a column to indicate whether the prediction was correct or not
            model_predictions['correct'] = model_predictions.apply(
                lambda x: 1 if x['pred'] == x['actual'] else 0, axis=1)
            # Save model predictions to the class
            self.model_predictions = model_predictions
            # Save training data used to train the model
            self.training_data = X
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
                str(self.performance.get(main_performance_metric))))
        # Save the data used to train the model
        data_save_dir = os.path.join(project_dir, 'data', 'training_data', self.model_id)
        with open(data_save_dir, 'wb') as f_out:
            joblib.dump(self.training_data, f_out)
        self.training_data.to_csv()
        # Save the trained model, if requested
        if self.save_trained_model and not self.test_mode:
            self.save_model()
