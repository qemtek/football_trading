import logging
import os
import numpy as np
import pandas as pd
import datetime as dt
import joblib
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from configuration import model_dir
from src.utils.base_model import load_model
from src.utils.db import run_query, connect_to_db
from src.utils.xgboost import get_features, get_manager_features, \
    get_feature_data, get_manager, get_profit
from src.utils.team_id_functions import fetch_name

logger = logging.getLogger('XGBoostModel')


class Model:
    """Anything that can be used by all models goes in this class"""
    def __init__(self, test_mode=False, load_model=False,
                 load_model_date=None, save_trained_model=True,
                 upload_historic_predictions=None):
        self.params = None
        self.trained_model = None
        self.model_type = None
        self.model_id = None
        self.param_grid = None
        self.model_object = None
        self.performance_metrics = [accuracy_score]
        self.scoring = 'accuracy'
        # Test mode will subsample the data to make things faster and not
        # save any data to the sqlite3 database
        self.test_mode = test_mode
        # The class object of the model you want to use
        self.model_object = None
        # Somewhere to store the trained model
        self.trained_model = None
        # A list of performance metrics (pass the functions, they must
        # take actuals, predictions as the first and second arguments
        self.performance_metrics = [balanced_accuracy_score, accuracy_score]
        # A dictionary to store the performance metrics for the trained model
        self.performance = {}
        # The date this class was instantiated
        self.creation_date = str(dt.datetime.today().date())
        # A unique identifier for this model
        self.model_id = "{}_{}_{}".format(
            self.model_type, self.creation_date, str(abs(hash(dt.datetime.today()))))
        # How many games to go back when generating training data
        self.window_length = 8
        # Name of the target variable (or variables, stored in a list)
        self.target = ['full_time_result']
        # The metric used to evaluate model performance
        self.scoring = 'balanced_accuracy'
        self.upload_historic_predictions = upload_historic_predictions
        # The minimum date to get training data from
        self.min_training_data_date = '2013-08-01'
        # A list of features used in the model
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
        self.training_data_query = \
            """select t1.*, m_h.manager home_manager, m_h.start_date home_manager_start, 
            m_a.manager away_manager, m_a.start_date away_manager_start,
             b365_home_odds, b365_draw_odds, b365_away_odds 
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

        # Attempt to load a model
        load_successful = False
        if load_model:
            load_successful = self.load_model(model_type=self.model_type, date=load_model_date)

        # If load model is false or model loading was unsuccessful, train a new model
        if not any([load_model, load_successful]):
            logger.info("Training a new model.")
            df = self.get_training_data()
            X, y = self.get_data(df)
            X[self.model_features] = self.preprocess(X[self.model_features])
            self.optimise_hyperparams(X[self.model_features], y, param_grid=self.param_grid)
            self.train_model(X=X, y=y)
            if save_trained_model:
                self.save_model()

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
        model = self.model_object() if self.params is None else self.model_object(**self.params)
        clf = GridSearchCV(model, param_grid, verbose=1, scoring=self.scoring, n_jobs=1)
        clf.fit(X_train, y_train)
        # Train a second model using the default parameters
        clf2 = self.model_object() if self.params is None else self.model_object(**self.params)
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

    def predict(self, **kwargs):
        """Predict the outcome of a matchup, given the team id's and date"""
        info = self.get_info(
            home_id=int(kwargs.get('home_id')),
            away_id=int(kwargs.get('away_id')),
            date=str(pd.to_datetime(kwargs.get('date')).date()),
            season=str(kwargs.get('season')))
        # Predict using the predict method of the parent class
        X, _ = self.get_data(info)
        X = self.preprocess(X)
        preds = self.trained_model.predict_proba(X) if self.trained_model is not None else None
        # Return predictions
        output = {"H": round(preds[0][2], 2),
                  "D": round(preds[0][1], 2),
                  "A": round(preds[0][0], 2)}
        return output

    def preprocess(self, X):
        """Apply preprocessing steps to data"""
        return np.array(X)

    def get_training_data(self):
        pass

    def save_model(self):
        if self.trained_model is None:
            logger.error("Trying to save a model that is None, aborting.")
        else:
            # Save the model ID inside the model object (so we know which
            # model made which predictions in the DB)
            self.trained_model.model_id = self.model_id
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
            self.model_id = self.trained_model.model_id
            self.params = model.get_params()
            return True

    def train_model(self, X, y):
        """Train a model on 90% of the data and predict 10% using KFold validation,
        such that a prediction is made for all data"""
        logger.info("Training model.")
        kf = KFold(n_splits=10)
        model_predictions = pd.DataFrame()
        for train_index, test_index in kf.split(X):
            model = self.model_object().fit(
                X=np.array(X.iloc[train_index, :][self.model_features]),
                y=np.ravel(np.array(y.iloc[train_index])))
            predictions = model.predict(np.array(X.iloc[test_index, :][self.model_features]))
            actuals = y.iloc[test_index]
            model_predictions = model_predictions.append(
                pd.concat([
                    X.iloc[test_index, :],
                    pd.DataFrame(predictions, columns=['pred'], index=X.iloc[test_index, :].index),
                    actuals], axis=1))
            # Assess the model performance using the first performance metric
            main_performance_metric = self.performance_metrics[0].__name__
            performance = self.performance_metrics[0](actuals, predictions)
            # If the model performs better than the previous model, save it
            # ToDo: Returning 0 when there is no performance score only works
            #  for performance scores where higher is better
            if performance > self.performance.get(main_performance_metric, 0):
                self.trained_model = model
                for metric in self.performance_metrics:
                    metric_name = metric.__name__
                    self.performance[metric_name] = metric(actuals, predictions)
        # Upload the predictions to the model_predictions table
        conn, cursor = connect_to_db()
        # Add model ID so we can compare model performances
        model_predictions['model_id'] = self.model_id
        # Add profit made if we bet on the game
        model_predictions['profit'] = model_predictions.apply(lambda x: get_profit(x), axis=1)
        if (not self.test_mode) or self.upload_historic_predictions:
            model_predictions.to_sql(
                'historic_predictions', con=conn, if_exists='append')
        conn.close()

    def get_data(self, df):
        logger.info("Preprocessing data and generating features.")
        # Add on manager features
        df = get_manager_features(df)
        # Get team feature data (unprocessed)
        df2 = get_feature_data(self.min_training_data_date)
        # Filter out the first window_length and last game weeks from the data
        df = df[(df['fixture_id'] > self.window_length * 10) & (df['fixture_id'] < 370)]
        # Filter out games that had red cards
        # ToDo: Test whether removing red card games is beneficial
        # df = df[(df['home_red_cards'] == 0) & (df['away_red_cards'] == 0)]
        identifiers = ['fixture_id', 'date', 'home_team', 'home_id',
                       'away_team', 'away_id', 'season']
        # If in test mode, only calculate the first 100 rows
        if self.test_mode and len(df) > 100:
            df = df.sample(100)
        # Generate features for each fixture
        y = pd.DataFrame()
        X = pd.DataFrame()
        for i in range(len(df)):
            row = df.iloc[i, :]
            features = get_features(
                row=row,
                index=df.index[i],
                team_data=df2,
                identifiers=identifiers,
                window_length=self.window_length
            )
            X = X.append(features)
            if self.target[0] in row.index:
                y = y.append(row[self.target])
        return X, y

    def get_historic_predictions(self):
        conn, cursor = connect_to_db()
        df = run_query(
            cursor, "select * from historic_predictions where "
                    "model_id = '{}'".format(self.model_id))
        return df

    @staticmethod
    def get_info(home_id, away_id, date, season):
        """Given the data and home/away team id's, get model features"""
        h_manager = get_manager(team_id=home_id, date=date)
        a_manager = get_manager(team_id=away_id, date=date)
        # Check that data was retrieved (catch the error sooner to speed up debugging)
        assert len(h_manager) > 0, 'No data returned for home manager'
        assert len(a_manager) > 0, 'No data returned for away manager'
        # Get the max date from the database
        conn, cursor = connect_to_db()
        max_date = run_query(cursor, 'select max(date) from main_fixtures')
        max_date = pd.to_datetime(max_date.iloc[0, 0])
        # set the fixture_id to be 1 higher than the max fixture_id for that season
        max_fixture = run_query(
            cursor,
            "select max(fixture_id) id from main_fixtures "
            "where date = '{}'".format(str(max_date)))
        max_fixture = max_fixture.iloc[0, 0]
        info_dict = {
            "date": date,
            "home_id": home_id,
            "home_team": fetch_name(home_id),
            "away_id": away_id,
            "away_team": fetch_name(away_id),
            "fixture_id": max_fixture,
            "home_manager_start": h_manager.loc[0, "start_date"],
            "away_manager_start": a_manager.loc[0, "start_date"],
            "season": season
        }
        output = pd.DataFrame()
        output = output.append(pd.DataFrame(info_dict, index=[0]))
        conn.close()
        return output

    def get_training_data(self):
        conn, cursor = connect_to_db()
        # Get all fixtures after game week 8, excluding the last game week
        df = run_query(cursor, self.training_data_query)
        return df

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
