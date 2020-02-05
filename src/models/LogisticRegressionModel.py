import numpy as np
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base_model import Model

logger = logging.getLogger("LogisticRegressionModel")


class LogisticRegressionModel(Model):
    """Everything specific to SKLearn models goes in this class"""

    def __init__(
        self,
        test_mode=False,
        load_model=False,
        load_model_date=None,
        save_trained_model=True,
        upload_historic_predictions=None,
    ):
        # Call the __init__ method of the parent class
        self.model_object = LogisticRegression
        # The name of the model you want ot use
        self.model_type = self.model_object.__name__
        # Initial model parameters (without tuning)
        self.params = {"max_iter": 1000000}
        # Define a grid for hyper-parameter tuning
        self.param_grid = {
            "penalty": ["l2", "none"],
            "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        }
        super().__init__(
            test_mode=test_mode,
            load_model=load_model,
            load_model_date=load_model_date,
            save_trained_model=save_trained_model,
            upload_historic_predictions=upload_historic_predictions,
        )

    def preprocess(self, X):
        """Standardise the data and return the result"""
        # Remove NaN
        X = X.dropna()
        # Convert to np array
        X = np.array(X)
        # Standardize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X


if __name__ == "__main__":
    model = LogisticRegressionModel(save_trained_model=True)
