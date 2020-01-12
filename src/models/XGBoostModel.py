import xgboost as xgb
import numpy as np
import os
import pandas as pd
import logging
import datetime as dt
import joblib

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from src.models.base_model import Model
from src.utils.db import run_query, connect_to_db
from src.utils.xgboost import get_features
from configuration import model_dir

logger = logging.getLogger('XGBoostModel')


class XGBoostModel(Model):
    """Everything specific to the XGBoost goes in this class"""
    def __init__(self, test_mode=False, load_model=False, load_model_date=None):
        # Call the __init__ method of the parent class
        super().__init__()
        self.test_mode = test_mode
        self.model_type = 'XGBoost'
        self.param_grid = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}
        self.params = {'n_estimators': 100}
        self.model = None
        self.accuracy = None
        self.balanced_accuracy = None
        self.model_features = [
            'avg_goals_for_home',
            'avg_goals_against_home',
            'sd_goals_for_home',
            'sd_goals_against_home',
            'avg_shots_for_home',
            'avg_shots_against_home',
            'sd_shots_for_home',
            'sd_shots_against_home',
            'avg_yellow_cards_home',
            'avg_red_cards_home',
            'b365_win_odds_home',
            'avg_perf_vs_bm_home',
            'manager_new_home',
            'manager_age_home',
            'win_rate_home',
            'draw_rate_home',
            'loss_rate_home',
            'home_advantage_sum_home',
            'home_advantage_avg_home',
            'avg_goals_for_away',
            'avg_goals_against_away',
            'sd_goals_for_away',
            'sd_goals_against_away',
            'avg_shots_for_away',
            'avg_shots_against_away',
            'sd_shots_for_away',
            'sd_shots_against_away',
            'avg_yellow_cards_away',
            'avg_red_cards_away',
            'b365_win_odds_away',
            'avg_perf_vs_bm_away',
            'manager_new_away',
            'manager_age_away',
            'win_rate_away',
            'draw_rate_away',
            'loss_rate_away',
            'home_advantage_sum_away',
            'home_advantage_avg_away'
        ]
        self.window_length = 8
        self.target = ['full_time_result']
        self.balanced_accuracy = 0
        self.scoring = 'balanced_accuracy'

        # Attempt to load a model
        load_successful = False
        if load_model:
            load_successful = self.load_model(model_type=self.model_type, date=load_model_date)

        # If load model is false or model loading was unsuccessful, train a new model
        if any([load_model, load_successful]):
            logger.info("Training a new model.")
            df = self.get_training_data()
            self.ids, self.X, self.y = self.preprocess(df)
            self.optimise_hyperparams(self.X[self.model_features], self.y)
            self.train_model(self.X[self.model_features], self.y)

    def preprocess(self, df):
        logger.info("Preprocessing data and generating features.")
        # Get manager features (time as manager) (manager age is logged to reduce scale)
        df['date'] = pd.to_datetime(df['date'])
        df['home_manager_start'] = pd.to_datetime(df['home_manager_start'])
        df['home_manager_age'] = df.apply(
            lambda x: np.log10(round((x['date'] - x['home_manager_start']).days)), axis=1)
        df['away_manager_start'] = pd.to_datetime(df['away_manager_start'])
        df['away_manager_age'] = df.apply(
            lambda x: np.log10(round((x['date'] - x['away_manager_start']).days)), axis=1)
        df['home_manager_new'] = df['home_manager_age'].apply(lambda x: 1 if x <= 70 else 0)
        df['away_manager_new'] = df['away_manager_age'].apply(lambda x: 1 if x <= 70 else 0)
        # Get team stats
        conn, cursor = connect_to_db()
        df2 = run_query(cursor, "select * from team_fixtures where date > '2013-08-01'")
        df2['date'] = pd.to_datetime(df2['date'])
        df2 = pd.merge(
            df2,
            df[['date', 'season', 'fixture_id', 'home_manager_age', 'away_manager_age',
                'home_manager_new', 'away_manager_new']],
            on=['date', 'season', 'fixture_id'])
        # Filter out the first window_length and last game weeks from the data
        df = df[(df['fixture_id'] > self.window_length * 10) & (df['fixture_id'] < 370)]
        # Filter out games that had red cards
        df = df[(df['home_red_cards'] == 0) & (df['away_red_cards'] == 0)]
        # If in test mode, only calculate the first 100 rows
        if self.test_mode:
            df = df.sample(100)
        y = pd.DataFrame()
        X = pd.DataFrame()
        for i in range(len(df)):
            row = df.iloc[i, :]
            home_features = get_features(
                row,
                team_data=df2,
                type='home',
                window_length=self.window_length,
                index=df.index[i])
            away_features = get_features(
                row,
                team_data=df2,
                type='away',
                window_length=self.window_length,
                index=df.index[i])
            features = pd.concat([home_features, away_features], axis=1)
            X = X.append(features)
            y = y.append(row[self.target])
        # Create an ID object to identify predictions later (if needed)
        ids = df[['fixture_id', 'date', 'home_team', 'home_id', 'away_team', 'away_id']]
        return ids, X, y

    def train_model(self, X, y):
        """Train a model on 90% of the data and predict 10% using KFold validation,
        such that a prediction is made for all data"""
        logger.info("Training model.")
        kf = KFold(n_splits=10)
        model_predictions = pd.DataFrame()
        for train_index, test_index in kf.split(X):
            xgb_model = xgb.XGBClassifier().fit(
                X=np.array(X.iloc[train_index, :]),
                y=np.array(y.iloc[train_index]))
            predictions = xgb_model.predict(np.array(X.iloc[test_index, :]))
            actuals = y.iloc[test_index]
            model_predictions = model_predictions.append(
                pd.concat([
                    X.iloc[test_index, :],
                    pd.DataFrame(predictions, columns=['pred'], index=X.iloc[test_index, :].index),
                    actuals], axis=1))
            balanced_accuracy = balanced_accuracy_score(actuals, predictions)
            accuracy = accuracy_score(actuals, predictions)
            # If the model performs better than the previous model, save it
            if balanced_accuracy > self.balanced_accuracy:
                self.model = xgb_model
                self.balanced_accuracy = balanced_accuracy
                self.accurracy = accuracy
            # Save the model predictions to the class
            self.predictions = model_predictions

    def optimise_hyperparams(self, X, y, param_grid=None):
        logger.info("Optimising hyperparameters")
        param_grid = self.param_grid if param_grid is None else param_grid
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        xgb_model = xgb.XGBClassifier()
        clf = GridSearchCV(xgb_model, param_grid, verbose=1, scoring=self.scoring)
        clf.fit(X_train, y_train)
        # Train a second model using the default parameters
        clf2 = xgb.XGBClassifier(params=self.params)
        clf2.fit(X_train, y_train)
        # Compare these params to existing params. If they are better, use them.
        clf_performance = balanced_accuracy_score(y_test, clf.best_estimator_.predict(X_test))
        clf2_performance = balanced_accuracy_score(y_test, clf2.predict(X_test))
        if clf2_performance > clf_performance:
            logger.info("Hyperparameter optimisation improves on previous model, "
                        "saving hyperparameters.")
            self.params = clf.best_params_

    def save_model(self):
        if self.model is None:
            logger.error("Trying to save a model that is None, aborting.")
        else:
            save_dir = os.path.join(
                model_dir, self.model_type+str(dt.datetime.today().date())+'.joblib')
            logger.info("Saving model to {} with joblib.".format(save_dir))
            joblib.dump(self.model, open(save_dir, "wb"))

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.model.predict_proba(X)

    def predict(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)


if __name__ == '__main__':
    model = XGBoostModel()
