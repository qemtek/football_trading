from src.models.templates.XGBoostModel import XGBoostModel
from src.utils.logging import get_logger
from src.utils.xgboost import get_team_model_performance, upload_to_table, get_profit2

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
                                   model_id = 'XGBClassifier_2020-02-09_7621640869568470400'"""

        self.target = 'model1_correct'

        # A list of features used in the model
        self.model_features = [
            'b365_home_odds',
            'b365_draw_odds',
            'b365_away_odds',
            'predict_proba_H',
            'predict_proba_D',
            'predict_proba_A',
            'home_model_perf',
            'away_model_perf',
            'bookmaker_odds_dif',
            'model_odds_dif',

            'model_bookmaker_odds_dif',
        ]
        df = self.get_training_data()
        df = df.rename(
            {'actual': 'model1_actual', 'pred': 'model1_pred',
             'profit': 'model1_profit', 'correct': 'model1_correct'}, axis=1)
        y = df[self.target]
        X = df.drop([self.target], axis=1)
        X = self.preprocess(X)
        self.train_model(X=X, y=y)
        self.model_predictions['profit'] = self.model_predictions.apply(
            lambda x: get_profit2(x), axis=1)
        self.model_predictions['correct'] = self.model_predictions.apply(
            lambda x: 1 if x['pred'] == x['actual'] else 0, axis=1)

        if upload_predictions:
            upload_to_table(
                self.model_predictions,
                table_name='historic_predictions2',
                model_id=self.model_id)

    def preprocess(self, X):
        X['bookmaker_odds_dif'] = X['b365_home_odds'] - X['b365_away_odds']
        X['model_odds_dif'] = X['predict_proba_H'] - X['predict_proba_A']
        X['model_bookmaker_odds_dif'] = X['bookmaker_odds_dif'] - X['model_odds_dif']
        X['home_model_perf'] = X.apply(
            lambda x: get_team_model_performance(x, self.previous_model_id, True), axis=1)
        X['away_model_perf'] = X.apply(
            lambda x: get_team_model_performance(x, self.previous_model_id, False), axis=1)
        return X


if __name__ == '__main__':
    model = BetOrNoBet(problem_name='bet_or_no_bet', save_trained_model=True, upload_predictions=True)
