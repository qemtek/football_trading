import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score

from football_trading.src.models.templates.sklearn_model import SKLearnModel


class XGBoostModel(SKLearnModel):
    """Everything specific to SKLearn models goes in this class"""
    def __init__(self,
                 test_mode=False,
                 save_trained_model=True,
                 load_trained_model=False,
                 load_model_date=None,
                 compare_models=False,
                 problem_name=None,
                 is_classifier=True):
        self.model_object = xgb.XGBClassifier if is_classifier else xgb.XGBRegressor
        super().__init__(model_object=self.model_object,
                         test_mode=test_mode,
                         save_trained_model=save_trained_model,
                         load_trained_model=load_trained_model,
                         load_model_date=load_model_date,
                         compare_models=compare_models,
                         problem_name=problem_name)
        self.performance_metrics = [accuracy_score, balanced_accuracy_score] \
            if is_classifier else [mean_squared_error, r2_score]
        # Initial model parameters (without tuning)
        self.params = {'n_estimators': 100}
        # Define a grid for hyper-parameter tuning
        self.param_grid = {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}

    def get_shap_explainer(self, X, plot_force=False):
        """Explain model predictions using SHAP"""
        explainer = shap.TreeExplainer(self.trained_model)
        shap_values = explainer.shap_values(X)
        # visualize the first prediction's explanation
        # (use matplotlib=True to avoid Javascript)
        if plot_force:
            shap.force_plot(explainer.expected_value[0], shap_values[0])
        return explainer, shap_values


if __name__ == '__main__':
    model = XGBoostModel(test_mode=True)
