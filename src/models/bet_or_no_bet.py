from src.models.templates.XGBoostModel import XGBoostModel
from src.utils.base_model import get_logger
from src.utils.xgboost import get_team_model_performance, upload_to_table

logger = get_logger()


class BetOrNoBet(XGBoostModel):
    def __init__(self,
                 upload_predictions=False,
                 test_mode=False,
                 save_trained_model=True,
                 load_trained_model=False,
                 load_model_date=None,
                 problem_name=None
                 ):
        super().__init__(test_mode=test_mode,
                         save_trained_model=save_trained_model,
                         load_trained_model=load_trained_model,
                         load_model_date=load_model_date,
                         problem_name=problem_name)
        self.training_data_query = """select * from historic_predictions where \
                                   model_id = 'XGBClassifier_2020-02-09_3894820684076999804'"""
        self.target = 'correct'
        # A list of features used in the model
        self.model_features = [
            'b365_home_odds',
            'b365_draw_odds',
            'b365_away_odds',
            'predict_proba_H',
            'predict_proba_D',
            'predict_proba_A',
            'home_model_perf',
            'away_model_perf'
        ]
        df = self.get_training_data()
        y = df[self.target]
        X = df.drop(self.target, axis=1)
        X = self.preprocess(X)
        self.train_model(X=X, y=y)
        if upload_predictions:
            upload_to_table(
                self.model_predictions,
                table_name='historic_predictions',
                model_id=self.model_id)

    def preprocess(self, X):
        X['bookmaker_odds_dif'] = X['b365_win_odds_home'] - X['b365_win_odds_away']
        X['model_odds_dif'] = X['predict_proba_H'] - X['predict_proba_A']
        X['home_model_perf'] = X.apply(
            lambda x: get_team_model_performance(x, self.model_id, True), axis=1)
        X['away_model_perf'] = X.apply(
            lambda x: get_team_model_performance(x, self.model_id, False), axis=1)
        return X


if __name__ == '__main__':
    model = BetOrNoBet(test_mode=True, problem_name='bet_or_no_bet')
