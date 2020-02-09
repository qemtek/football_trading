import xgboost as xgb
import shap

from src.models.templates.sklearn_model import SKLearnModel
from src.utils.base_model import get_logger

logger = get_logger()


class XGBoostModel(SKLearnModel):
    """Everything specific to SKLearn models goes in this class"""
    def __init__(self,
                 test_mode=False,
                 save_trained_model=True,
                 load_trained_model=False,
                 load_model_date=None,
                 problem_name=None):
        super().__init__(
            test_mode=test_mode,
            model_object=xgb.XGBClassifier,
            save_trained_model=save_trained_model,
            load_trained_model=load_trained_model,
            load_model_date=load_model_date,
            problem_name=problem_name
        )
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
