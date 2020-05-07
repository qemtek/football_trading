import xgboost as xgb
import shap

from football_trading.src.models.templates.sklearn_model import SKLearnModel
from football_trading.src.utils.logging import get_logger

logger = get_logger()


class XGBoostModel(SKLearnModel):
    """Everything specific to SKLearn models goes in this class"""
    def __init__(self,
                 test_mode=False,
                 save_trained_model=True,
                 load_trained_model=False,
                 load_model_date=None,
                 compare_models=False,
                 problem_name=None,
                 is_classifier=True,
                 local=True):
        self.model_object = xgb.XGBClassifier if is_classifier else xgb.XGBRegressor
        super().__init__(model_object=self.model_object,
                         test_mode=test_mode,
                         save_trained_model=save_trained_model,
                         load_trained_model=load_trained_model,
                         load_model_date=load_model_date,
                         compare_models=compare_models,
                         problem_name=problem_name,
                         is_classifier=is_classifier,
                         local=local)
        # Initial model parameters (without tuning)
        self.params = {'n_estimators': 100}
        # Define a grid for hyper-parameter tuning
        self.param_grid = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}

    def get_shap_values(self, X):
        # Explain the model's predictions using SHAP
        explainer = shap.TreeExplainer(self.trained_model)
        shap_values = explainer.shap_values(X)
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        # shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
        return shap_values

if __name__ == '__main__':
    model = XGBoostModel(test_mode=True)
