import datetime as dt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.templates.sklearn_model import SKLearnModel
from src.utils.base_model import get_logger

logger = get_logger()


class LogisticRegressionModel(SKLearnModel):
    """Everything specific to LogisticRegression models goes in this class"""

    def __init__(
        self,
        test_mode=False,
        load_model=False,
        load_model_date=None,
        save_trained_model=True,
        upload_historic_predictions=None,
    ):
        # Call the __init__ method of the parent class
        super().__init__(
            test_mode=test_mode,
            upload_historic_predictions=upload_historic_predictions,
        )
        # Define the model object
        self.model_object = LogisticRegression
        # The name of the model you want ot use
        self.model_type = self.model_object.__name__
        # A unique identifier for this model
        self.model_id = "{}_{}_{}".format(
            self.model_type, self.creation_date, str(abs(hash(dt.datetime.today()))))
        # Initial model parameters (without tuning)
        self.params = {"max_iter": 100000}
        # Define a grid for hyper-parameter tuning
        self.param_grid = {
            "penalty": ["l2", "none"],
            "solver": ["newton-cg", "lbfgs", "sag", "saga"],
        }
        # Attempt to load a model
        load_successful = False
        if load_model:
            load_successful = self.load_trained_model(model_type=self.model_type, date=load_model_date)
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

    def preprocess(self, X):
        """Standardise the data and return the result"""
        # Remove NaN
        X = X.dropna()
        # Standardize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X


if __name__ == "__main__":
    model = LogisticRegressionModel(save_trained_model=True)
