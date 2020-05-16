import pandas as pd
import numpy as np
import datetime as dt
import sqlite3

from sklearn.utils.class_weight import compute_class_weight

from football_trading.src.models.templates.XGBoostModel import XGBoostModel
from football_trading.src.utils.general import suspend_logging, time_function
from football_trading.src.utils.db import run_query, connect_to_db
from football_trading.src.utils.xgboost import get_features, get_manager_features, \
    get_feature_data, get_manager, get_profit, get_profit_betting_on_fav, apply_profit_weight
from football_trading.src.utils.team_id_functions import fetch_name
from football_trading.settings import sql_dir, data_dir, plots_dir, LOCAL, DB_DIR, S3_BUCKET_NAME
from football_trading.src.utils.logging import get_logger
from football_trading.src.utils.s3_tools import upload_to_s3

logger = get_logger()


class MatchResultXGBoost(XGBoostModel):
    """Everything specific to SKLearn models goes in this class
    """
    def __init__(self, test_mode=False, load_trained_model=False, load_model_date=None,
                 save_trained_model=True, upload_historic_predictions=None, apply_sample_weight=False,
                 compare_models=False, problem_name='match_predict', production_model=True, create_plots=True,
                 optimise_hyperparams=False):
        super().__init__(test_mode=test_mode, save_trained_model=save_trained_model,
                         load_trained_model=load_trained_model, load_model_date=load_model_date,
                         compare_models=compare_models,  problem_name=problem_name, local=LOCAL)
        # Random seed for reproducibility
        self.random_seed = 42
        self.apply_sample_weight = apply_sample_weight
        self.upload_historic_predictions = upload_historic_predictions
        # Initial model parameters (without tuning)
        self.params = {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.01}
        # Define a grid for hyper-parameter tuning
        self.param_grid = {'max_depth': [2, 5, 10, 15], 'n_estimators': [100, 500, 1000],
                           'learning_rate': [0.01, 0.1, 0.2]}
        # The minimum date to get training data from
        self.min_training_data_date = '2012-08-01'
        # How many games to go back when generating training data
        self.window_length = 8
        # Name of the target variable (or variables, stored in a list)
        self.target = ['full_time_result']
        # The metric used to evaluate model performance
        self.scoring = 'roc_auc_ovo'
        # Store the predictions made by the trained model here
        self.model_predictions = None
        # Load model if requested
        self.trained_model = self.load_model(load_model_date) if load_trained_model else None
        # A list of features used in the model
        self.model_features = [
            # Aggregated stats home team
            'avg_goals_for_home', 'avg_goals_against_home', 'avg_goals_for_ha_home', 'avg_goals_against_ha_home',
            'sd_goals_for_home', 'sd_goals_against_home', 'avg_shots_for_home', 'avg_shots_against_home',
            'sd_shots_for_home', 'sd_shots_against_home', 'avg_yellow_cards_home', 'avg_red_cards_home',
            'avg_perf_vs_bm_home', 'manager_new_home', 'manager_age_home', 'win_rate_home', 'draw_rate_home',
            'loss_rate_home', 'home_advantage_sum_home', 'home_advantage_avg_home',
            # Aggregated stats away team
            'avg_goals_for_away',
            'avg_goals_against_away', 'avg_goals_for_ha_away', 'avg_goals_against_ha_away', 'sd_goals_for_away',
            'sd_goals_against_away', 'avg_shots_for_away', 'avg_shots_against_away', 'sd_shots_for_away',
            'sd_shots_against_away', 'avg_yellow_cards_away', 'avg_red_cards_away', 'avg_perf_vs_bm_away',
            'manager_new_away', 'manager_age_away', 'win_rate_away', 'draw_rate_away', 'loss_rate_away',
            'home_advantage_sum_away', 'home_advantage_avg_away',
            # Match odds (bet365)
            'home_odds', 'draw_odds',  'away_odds',
            # Last 5 game stats home team
            'goals_for_l1_home', 'goals_for_l2_home','goals_for_l3_home','goals_for_l4_home','goals_for_l5_home',
            'goals_against_l1_home', 'goals_against_l2_home','goals_against_l3_home','goals_against_l4_home',
            'goals_against_l5_home', 'goal_difference_l1_home', 'goal_difference_l2_home', 'goal_difference_l3_home',
            'goal_difference_l4_home', 'goal_difference_l5_home', 'shots_for_l1_home', 'shots_for_l2_home',
            'shots_for_l3_home', 'shots_for_l4_home', 'shots_for_l5_home', 'shots_against_l1_home',
            'shots_against_l2_home', 'shots_against_l3_home', 'shots_against_l4_home', 'shots_against_l5_home',
            'shot_difference_l1_home', 'shot_difference_l2_home', 'shot_difference_l3_home', 'shot_difference_l4_home',
            'shot_difference_l5_home',
            'is_home_l1_home', 'is_home_l2_home', 'is_home_l3_home', 'is_home_l4_home', 'is_home_l5_home',
            # Last 5 game stats away team
            'goals_for_l1_away', 'goals_for_l2_away', 'goals_for_l3_away', 'goals_for_l4_away', 'goals_for_l5_away',
            'goals_against_l1_away', 'goals_against_l2_away', 'goals_against_l3_away', 'goals_against_l4_away',
            'goals_against_l5_away', 'goal_difference_l1_away', 'goal_difference_l2_away', 'goal_difference_l3_away',
            'goal_difference_l4_away', 'goal_difference_l5_away', 'shots_for_l1_away', 'shots_for_l2_away',
            'shots_for_l3_away', 'shots_for_l4_away', 'shots_for_l5_away', 'shots_against_l1_away',
            'shots_against_l2_away', 'shots_against_l3_away', 'shots_against_l4_away', 'shots_against_l5_away',
            'shot_difference_l1_away', 'shot_difference_l2_away', 'shot_difference_l3_away', 'shot_difference_l4_away',
            'shot_difference_l5_away',
            'is_home_l1_away', 'is_home_l2_away', 'is_home_l3_away', 'is_home_l4_away', 'is_home_l5_away',
        ]
        # Specify the query itself, or the location of the query to retrieve the training data
        self.training_data_query = f"{sql_dir}/get_training_data.sql"
        # Train a model if one was not loaded
        if self.trained_model is None:
            logger.info("Training a new model.")
            X, y = self.get_training_data()
            X[self.model_features] = self.preprocess(X[self.model_features])
            # Apply sample weights (if requsted)
            if self.apply_sample_weight:
                #sample_weight = np.array(1/abs(X['avg_goals_for_home'] - X['avg_goals_for_away']))
                # td = X
                # td['target'] = y
                # sample_weight = np.array(td.apply(lambda x: apply_profit_weight(x), axis=1))
                weights = compute_class_weight('balanced', np.unique(y), y['full_time_result'])
                class_weights = {'A': weights[0], 'D': weights[1], 'H': weights[2]}
                sample_weight = y['full_time_result'].apply(lambda x: class_weights.get(x))
            else:
                sample_weight = np.ones(len(X))
            # Optimise hyper-parameters using a grid search
            if optimise_hyperparams:
                self.optimise_hyperparams(X=X, y=y, param_grid=self.param_grid)
            # Train the model
            self.train_model(X=X, y=y, sample_weight=sample_weight)
            # Compare model performance vs the latest model, save model data
            # if the new model has a better performance
            model_improves = True if not self.compare_models else self.compare_latest_model()
            # Save the trained model, if requested
            if self.save_trained_model and not test_mode:
                self.save_model(local=LOCAL, save_to_production=model_improves)
            # Add profit made if we bet on the game
            self.model_predictions['profit'] = self.model_predictions.apply(
                lambda x: get_profit(x), axis=1)
            # Add profit made betting on the favourite
            self.model_predictions['profit_bof'] = self.model_predictions.apply(
                lambda x: get_profit_betting_on_fav(x), axis=1)
            # Upload predictions to the local db
            if upload_historic_predictions and not test_mode:
                upload_cols = ['fixture_id', 'home_team', 'away_team', 'season',
                               'date', 'pred', 'actual', 'profit', 'profit_bof']
                self.save_prediction_data(cols_to_save=upload_cols)
            # Upload the new DB to S3 if LOCAL is False
            if not LOCAL and not test_mode:
                upload_to_s3(local_path=f"{DB_DIR}", s3_path='db.sqlite', bucket=S3_BUCKET_NAME)
            # Create plots if requested
            if create_plots:
                logger.info('Generating plots')
                from football_trading.src.ModelEvaluator import ModelEvaluator
                training_data = self.training_data
                training_data['X_train'] = training_data['X_train'][self.model_features]
                training_data['X_test'] = training_data['X_test'][self.model_features]
                ModelEvaluator(training_data=training_data, trained_model=self.trained_model,
                               model_id=self.model_id, is_classifier=True, plots_dir=plots_dir,
                               data_dir=f"{data_dir}")

    def save_prediction_data(self, *, cols_to_save):
        self.upload_to_table(
            df=self.model_predictions[cols_to_save],
            table_name='historic_predictions',
            model_id=self.model_id)

    @time_function(logger=logger)
    def upload_to_table(self, df, table_name, model_id=None):
        # Upload the predictions to the model_predictions table
        conn = connect_to_db()
        if 'creation_time' not in df.columns:
            df['creation_time'] = dt.datetime.now()
        try:
            model_ids = run_query(query=f'select distinct model_id from {table_name}')
            model_ids = list(model_ids['model_id'].unique())
        except sqlite3.OperationalError  as e:
            model_ids = []
        if model_id in model_ids:
            logger.warning(f'Model id: {model_id} already exists in the table {table_name}!')
        # Add model ID so we can compare model performances
        if model_id is not None:
            df['model_id'] = model_id
        df.to_sql(table_name, con=conn, if_exists='append', index=False)
        conn.close()

    @time_function(logger=logger)
    def get_training_data(self):
        # Get all fixtures after game week 8, excluding the last game week
        df = run_query(query=self.training_data_query)
        # Change names of b365 odds
        df['home_odds'] = df['b365_home_odds']
        df['draw_odds'] = df['b365_draw_odds']
        df['away_odds'] = df['b365_away_odds']
        df = df.drop(['b365_home_odds', 'b365_draw_odds', 'b365_away_odds'], axis=1)
        X, y = self.get_data(df)
        return X, y

    @time_function(logger=logger)
    def get_data(self, df):
        logger.info("Pre-processing data and generating features.")
        # Add on manager features
        df = get_manager_features(df)
        # Get team feature data (unprocessed)
        df2 = get_feature_data(self.min_training_data_date)
        # Filter out the first window_length and last game weeks from the data
        df = df[(df['fixture_id'] > self.window_length * 10) & (df['fixture_id'] < 370)].reset_index(drop=True)
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
    def get_info(home_id, away_id, date, season, home_odds, draw_odds, away_odds):
        """Given the data and home/away team id's, get model features"""
        conn = connect_to_db()
        h_manager = get_manager(team_id=home_id, date=date)
        a_manager = get_manager(team_id=away_id, date=date)
        # Check that data was retrieved (catch the error sooner to speed up debugging)
        assert len(h_manager) > 0, 'No data returned for home manager'
        assert len(a_manager) > 0, 'No data returned for away manager'
        # Get the max date from the database
        max_date = run_query(query='select max(date) from main_fixtures')
        max_date = pd.to_datetime(max_date.iloc[0, 0])
        # set the fixture_id to be 1 higher than the max fixture_id for that season
        max_fixture = run_query(
            query="select max(fixture_id) id from main_fixtures where date = '{}'".format(str(max_date)))
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
            "season": season,
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds
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
            season=str(kwargs.get('season')),
            home_odds=float(kwargs.get('home_odds')),
            draw_odds=float(kwargs.get('draw_odds')),
            away_odds=float(kwargs.get('away_odds'))
        )
        # Predict using the predict method of the parent class
        X, _ = self.get_data(info)
        X[self.model_features] = self.preprocess(X[self.model_features])
        if self.trained_model is not None:
            logger.info(f"Input data: {X}")
            preds = self.trained_model.predict_proba(X[self.model_features])
            logger.info(f"Output data: {preds}")
        else:
            logger.info("self.trained_model is None, so cannot make predictions. Returning None")
            preds = None
        # Return predictions
        output = {"H": round(preds[0][2], 2),
                  "D": round(preds[0][1], 2),
                  "A": round(preds[0][0], 2)}
        return output

    @time_function(logger=logger)
    def get_historic_predictions(self):
        """Get predictions on historic data for a particular model"""
        df = run_query(query="select * from historic_predictions where "
                             "model_id = '{}'".format(self.model_id))
        return df


if __name__ == '__main__':
    model = MatchResultXGBoost(
        save_trained_model=True,
        upload_historic_predictions=True,
        problem_name='match-predict-base',
        apply_sample_weight=False,)
