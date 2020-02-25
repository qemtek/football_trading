import pandas as pd
import numpy as np

from src.models.templates.XGBoostModel import XGBoostModel
from src.utils.base_model import get_logger, suspend_logging
from src.utils.base_model import time_function
from src.utils.db import run_query, connect_to_db
from src.utils.xgboost import get_features, get_manager_features, \
    get_feature_data, get_manager, get_profit, upload_to_table, get_profit_betting_on_fav
from src.utils.team_id_functions import fetch_name

logger = get_logger()


class MatchResultXGBoost(XGBoostModel):
    """Everything specific to SKLearn models goes in this class"""
    def __init__(self,
                 test_mode=False,
                 load_trained_model=False,
                 load_model_date=None,
                 save_trained_model=True,
                 upload_historic_predictions=None,
                 problem_name='match_predict'):
        super().__init__(
            test_mode=test_mode,
            save_trained_model=save_trained_model,
            load_trained_model=load_trained_model,
            load_model_date=load_model_date,
            problem_name=problem_name
        )
        self.upload_historic_predictions = upload_historic_predictions
        # Initial model parameters (without tuning)
        self.params = {'n_estimators': 100}
        # Define a grid for hyper-parameter tuning
        self.param_grid = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}
        # The minimum date to get training data from
        self.min_training_data_date = '2013-08-01'
        # How many games to go back when generating training data
        self.window_length = 8
        # Name of the target variable (or variables, stored in a list)
        self.target = ['full_time_result']
        # The metric used to evaluate model performance
        self.scoring = 'balanced_accuracy'
        # Store the predictions made by the trained model here
        self.model_predictions = None
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

        # Train a model if one was not loaded
        if self.trained_model is None:
            logger.info("Training a new model.")
            X, y = self.get_training_data()
            X[self.model_features] = self.preprocess(X[self.model_features])
            self.optimise_hyperparams(X[self.model_features], y, param_grid=self.param_grid)
            self.train_model(X=X, y=y)
            # Add profit made if we bet on the game
            self.model_predictions['profit'] = self.model_predictions.apply(
                lambda x: get_profit(x), axis=1)
            # Add profit made betting on the favourite
            self.model_predictions['profit_bof'] = self.model_predictions.apply(
                lambda x: get_profit_betting_on_fav(x), axis=1)
            if upload_historic_predictions:
                upload_to_table(
                    self.model_predictions,
                    table_name='historic_predictions',
                    model_id=self.model_id)

    @time_function(logger=logger)
    def get_training_data(self):
        # Get all fixtures after game week 8, excluding the last game week
        df = run_query(self.training_data_query)
        X, y = self.get_data(df)
        return X, y

    @time_function(logger=logger)
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

    @staticmethod
    @time_function(logger=logger)
    def get_info(home_id, away_id, date, season):
        """Given the data and home/away team id's, get model features"""
        conn = connect_to_db()
        h_manager = get_manager(team_id=home_id, date=date)
        a_manager = get_manager(team_id=away_id, date=date)
        # Check that data was retrieved (catch the error sooner to speed up debugging)
        assert len(h_manager) > 0, 'No data returned for home manager'
        assert len(a_manager) > 0, 'No data returned for away manager'
        # Get the max date from the database
        max_date = run_query('select max(date) from main_fixtures')
        max_date = pd.to_datetime(max_date.iloc[0, 0])
        # set the fixture_id to be 1 higher than the max fixture_id for that season
        max_fixture = run_query(
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

    @suspend_logging(logger=logger)
    def predict(self, **kwargs):
        """Predict the outcome of a matchup, given the team id's and date"""
        info = self.get_info(
            home_id=int(kwargs.get('home_id')),
            away_id=int(kwargs.get('away_id')),
            date=str(pd.to_datetime(kwargs.get('date')).date()),
            season=str(kwargs.get('season')))
        # Predict using the predict method of the parent class
        X, _ = self.get_data(info)
        X[self.model_features] = self.preprocess(X[self.model_features])
        preds = self.trained_model.predict_proba(np.array(X[self.model_features])) \
            if self.trained_model is not None else None
        # Return predictions
        output = {"H": round(preds[0][2], 2),
                  "D": round(preds[0][1], 2),
                  "A": round(preds[0][0], 2)}
        return output

    @time_function(logger=logger)
    def get_historic_predictions(self):
        """Get predictions on historic data for a particular model"""
        df = run_query(
            "select * from historic_predictions where "
            "model_id = '{}'".format(self.model_id))
        return df


if __name__ == '__main__':
    model = MatchResultXGBoost(save_trained_model=True, upload_historic_predictions=True)
