import os
import datetime as dt
import joblib
import pandas as pd
import sqlite3

from sklearn.metrics import balanced_accuracy_score, accuracy_score, r2_score, mean_absolute_error

from football_trading.settings import model_dir, PROJECTSPATH, available_models, S3_BUCKET_NAME
from football_trading.src.utils.logging import get_logger
from football_trading.src.utils.general import safe_open, time_function
from football_trading.src.utils.s3_tools import upload_to_s3, list_files, download_from_s3

logger = get_logger()

# Add all columns that what steps to fill NA's (steps are executed sequentially in a list
FILL_NA_LAST_VALUE = {'feature': ['groupby_col']}
# Specify columns to fill NA's with mean, and what cols to group by (or None)
FILL_NA_MEAN = {}
# Specify columns to fill NA's with median, and what cols to group by (or None)
FILL_NA_MEDIAN = {}
# Specify the order of the operations
fill_na_order = [FILL_NA_LAST_VALUE, FILL_NA_MEAN, FILL_NA_MEDIAN]


class BaseModel:
    """Anything that can be used by any model goes in this class
    """
    def __init__(self,
                 model_object=None,
                 load_trained_model=True,
                 save_trained_model=True,
                 test_mode=False,
                 load_model_date=None,
                 problem_name=None,
                 compare_models=True,
                 is_classifier=False):
        # Store all arguments passed to __init__ inside the class,
        # so we know what they were later
        self.model_object = model_object
        self.df_removed = pd.DataFrame()
        self.load_trained_model = load_trained_model
        self.save_trained_model = save_trained_model
        self.test_mode = test_mode
        self.load_model_date = load_model_date
        self.problem_name = problem_name
        self.compare_models = compare_models
        self.trained_model_features = None
        # Dataframes to store any data removed before or after feature generation
        # Create a dataframe to store rows excluded before generating features
        self.df_removed = pd.DataFrame()
        # Create a dataframe to store rows excluded after generating features
        self.df_removed2 = pd.DataFrame()
        # The date this class was instantiated
        self.creation_date = str(dt.datetime.today().date())
        # The name of the model you want to use
        self.model_type = self.model_object.__name__
        # A unique identifier for this model
        self.model_id = "{}_{}_{}_{}".format(
            self.problem_name, self.model_type, self.creation_date, str(abs(hash(dt.datetime.today()))))
        self.trained_model = None
        # Load previous best model
        self.previous_model = self.load_model(load_model_attributes=False)
        # The name of all features in the model, specified as a list
        self.model_features = None
        # Model parameters
        self.params = None
        # A place to store predictions made by the model
        self.model_predictions = None
        # What name to give the problem the model is trying to solve
        self.problem_name = problem_name
        # A list of performance metrics (pass the functions, they must
        # take actuals, predictions as the first and second arguments
        self.performance_metrics = [accuracy_score, balanced_accuracy_score] \
            if is_classifier else [mean_absolute_error, r2_score]
        # A dictionary to store the performance metrics for the trained model
        self.performance = {}
        # Name of the target variable (or variables, stored in a list)
        self.target = None
        # A query used to retrieve training data
        self.training_data_query = None
        # Store names of categorical features
        self.categorical_features = None
        # Location of the database (if using sqlite)
        self.db_dir = None

    def get_training_data(self) -> pd.DataFrame:
        return self.run_query(self.training_data_query)

    def generate_features(self, X) -> pd.DataFrame:
        return NotImplemented

    def encode_cateogrical_features(self, X) -> pd.DataFrame:
        """One-hot encode categorical features and add them to the feature list
        """
        X = X.copy()
        # ToDo: We still need to convert bimodal categorical features to 1/0
        # Only encode categories that have more than 2 unique values
        old_categorical_features = []
        for c in self.categorical_features:
            col_vals = X[c].dropna().reset_index(drop=True)
            first_val = col_vals[0] if len(col_vals) > 0 else None
            if len(col_vals.unique()) > 2 or isinstance(first_val, str) or isinstance(first_val, bool):
                old_categorical_features.append(c)
        for feature in self.categorical_features:
            X_dummies = pd.get_dummies(X[feature], prefix=feature)
            # Get encoded features
            X = pd.concat([X.drop(feature, axis=1), X_dummies], axis=1)
            # Add encoded features to the feature list
            self.model_features = self.model_features + list(X_dummies.columns)
            self.categorical_features = self.categorical_features + list(X_dummies.columns)
        # Remove unencoded categorical features from the feature list
        self.model_features = [feature for feature in self.model_features
                               if feature not in old_categorical_features]
        self.categorical_features = [feature for feature in self.categorical_features
                               if feature not in old_categorical_features]
        return X

    def preprocess(self, X, apply_filters=True) -> pd.DataFrame:
        X = X.copy()
        return X

    def optimise_hyperparams(self, X, y, param_grid=None) -> None:
        return NotImplemented

    def train_model(self, X, y) -> None:
        return NotImplemented

    def predict(self, X, preprocess=True, return_data=False,
                save_predictions=False, print_predictions=False):  # ToDo: Find out what datatype is output
        """Predict on a new set of data
        """
        if preprocess:
            X = self.preprocess(X)

        preds = self.trained_model.predict(X[self.trained_model.model_features])

        if save_predictions:
            df_comb = pd.concat([X, pd.Series(preds)], axis=1)
            df_comb.columns = list(df_comb.columns[:-1]) + ['pred']
            pred_dir = f"{PROJECTSPATH}/data/predictions/{str(dt.datetime.today().date())}.joblib"
            if os.path.exists(pred_dir):
                with safe_open(pred_dir, 'rb') as f_in:
                    other_preds = joblib.load(pred_dir)
                    # Join predictions with other predictions made that day
                    df_comb = pd.concat([other_preds, df_comb], axis=0)
            # Save the predictions
            with safe_open(pred_dir, 'wb') as f_out:
                joblib.dump(df_comb, f_out)

        X = pd.concat([X, pd.Series(preds)], axis=1)
        X.columns = list(X.columns[:-1]) + ['pred']

        # Print predictions in logger
        if print_predictions:
            for row in X.iterrows():
                logger.info(f"Prediction made. \n"
                            f"Data: {row[1][['id'] + self.trained_model.model_features]}\n"
                            f"Prediction: {row[1]['pred']}")

        if return_data:
            return X
        else:
            return preds

    def save_training_data(self) -> None:
        return NotImplemented

    def save_prediction_data(self, *, cols_to_save) -> None:
        return NotImplemented

    @time_function(logger=logger)
    def save_model(self, save_to_production=False, attributes=None, local=True) -> None:
        """Save a trained model to the models directory
        """
        if self.trained_model is None:
            logger.error("There is no model to save, aborting.")
        else:
            # Save the model ID inside the model object (so we know which
            # model made which predictions in the DB)
            self.trained_model.model_id = self.model_id
            self.trained_model.model_features = self.model_features
            self.trained_model.performance_metrics = self.performance_metrics
            self.trained_model.performance = self.performance
            # Add additional attributes if required
            if attributes is not None:
                # ToDo: Eventually store all attributes in this dict
                setattr(self.trained_model, 'additional_attrs', attributes)
                for name, value in attributes.items():
                    logger.info(f'Added additional parameter {name} with value {value} to trained model '
                                f'object (stored in model.additional_attrs)')
            file_name = self.model_id + '.joblib'
            save_dir = os.path.join(model_dir, file_name)
            logger.info("Saving model to {} with joblib.".format(save_dir))
            with safe_open(save_dir, "wb") as f_out:
                joblib.dump(self.trained_model, f_out)
            if local:
                if save_to_production:
                    save_dir = os.path.join(model_dir, 'in_production', file_name)
                    with safe_open(save_dir, "wb") as f_out:
                        joblib.dump(self.trained_model, f_out)
            else:
                upload_to_s3(save_dir, f'models/{self.model_id}.joblib', bucket=S3_BUCKET_NAME)
                logger.info(f'Model saved to S3 ({S3_BUCKET_NAME}/models/{self.model_id}.joblib')
                if save_to_production:
                    if not local:
                        upload_to_s3(save_dir, f'models/in_production/{self.model_id}.joblib',
                                     bucket=S3_BUCKET_NAME)

    @time_function(logger=logger)
    def load_model(self, date=None, load_model_attributes=True, local=True) -> None:
        """Load model from the local filesystem
        """
        keyword = self.problem_name
        # Check that the requested model type is one that we have
        if self.model_type not in available_models:
            logger.error("load_model: model_type must be one of {}. "
                         "Returning None".format(', '.join(available_models)))
            return None
        # Check if the supplied date is in the correct format
        model_date = date if date is not None else 'latest'
        logger.info("Attempting to load {} model for date: "
                    "{}".format(self.model_type, model_date))
        # Load all model directories
        if local:
            models = os.listdir(model_dir)
        else:
            s3_model_info = list_files(prefix='models', bucket=S3_BUCKET_NAME)
            models = [m.get('Key') for m in s3_model_info]
        # Filter for the model type
        models_filtered = [model for model in models if model.find(self.model_type) != -1]
        # Filter for keyword
        if keyword is not None:
            models_filtered = [model for model in models if model.split('_')[0] == keyword]
        # Filter for the date (if requested)
        if date is not None:
            models_filtered = [model for model in models if model.find(date) != -1]
        # Order models by date and pick the latest one
        if len(models_filtered) > 0:
            models_filtered.sort(reverse=True)
            model_name = models_filtered[0]
            logger.info("load_model: Loading model with filename: {}".format(model_name))
            if not local:
                download_from_s3(f"{model_dir}/{model_name}", s3_path=f"football-trading/models/{model_name}",
                                 bucket=S3_BUCKET_NAME)
            with safe_open(os.path.join(model_dir, model_name), "rb") as f_in:
                model = joblib.load(f_in)
        else:
            logger.warning('No available models of type {}.'.format(self.model_type))
            return None
        if load_model_attributes:
            # If additional attributes are stored, unwrap them and set as class attributes
            # ToDo: Eventually store all additional attributes in that dict
            if hasattr(model, 'additional_attrs'):
                for key, value in model.additional_attrs.items():
                    setattr(self, key, value)
            # Set the attributes of the model to those of the class
            logger.info(f'Loaded model {model.model_id}')
            self.model_id = model.model_id
            self.trained_model = model
            # Load training data
            logger.info('Loading training data')
            model_data_dir = f"{PROJECTSPATH}/data/training_data/{self.model_id}.joblib"
            if os.path.exists(model_data_dir):
                logger.info('Loading model training data')
                model_data = joblib.load(model_data_dir)
                self.training_data = model_data.get('train_test_data')
                self.df_removed = model_data.get('removed_data_without_features')
                self.df_removed2 = model_data.get('removed_data_with_features')
            # Load other parameters
            if hasattr(model, 'params'):
                self.params = model.params
            if hasattr(model, 'model_features'):
                self.trained_model_features = model.model_features
            if hasattr(model, 'performance_metrics'):
                self.performance_metrics = model.performance_metrics
            if hasattr(model, 'performance'):
                self.performance = model.performance
            else:
                logger.warning('The loaded model has no get_params method, cannot load model parameters.')
        return model

    @time_function(logger=logger)
    def compare_latest_model(self) -> bool:
        """Compare the newly trained model with the previous model of the same name (if one exists).
            Return True if the new model performs best, otherwise return False """

        main_performance_metric = self.performance_metrics[0].__name__
        new_performance = self.performance.get(main_performance_metric)
        old_performance = self.previous_model.performance.get(main_performance_metric)
        if new_performance > old_performance:
            logger.info('New model beats previous model. Replacing this model')
            logger.info('{}: Previous Model: {}, New Model: {}'.format(
                main_performance_metric, old_performance, new_performance
            ))
            return True
        else:
            logger.info('New model does not beat previous model.')
            logger.info('{}: Previous Model: {}, New Model: {}'.format(
                main_performance_metric, old_performance, new_performance
            ))
            return False

    def connect_to_db(self, path_to_db=None, check_same_thread=True):
        """# Connect to local sqlite3 database
        """
        # If no name is supplied, use the default name
        sqlite_file = self.db_dir if path_to_db is None else path_to_db
        # Establish a connection to the database
        conn = sqlite3.connect(sqlite_file, check_same_thread=check_same_thread, timeout=10)
        # Return the connection object
        return conn

    def run_query(self, query, params=None, return_data=True, path_to_db=None) -> pd.DataFrame:
        """Function to run a query on the DB while still keeping the column names. Returns a DataFrame
        """
        if os.path.exists(query):
            with open(query, 'r') as f_in:
                query = ''
                for line in f_in.readlines():
                    query = query + line
        with self.connect_to_db(path_to_db) as conn:
            cursor = conn.cursor()
            split_query = query.split(';')
            if len(split_query) > 1:
                for query in split_query:
                    # Run query
                    print('Multiple queries detected, not returning data')
                    cursor.execute(query, params if params is not None else [])
            else:
                # Run query
                cursor.execute(query, params if params is not None else [])
                # Get column names and apply to the data frame
                if return_data:
                    names = cursor.description
                    name_list = []
                    for name in names:
                        name_list.append(name[0])
                    # Convert the result into a DataFrame and add column names
                    df = pd.DataFrame(cursor.fetchall(), columns=name_list)
                    return df

    def fill_na_values(self, df):
        for operation in fill_na_order:
            for feature, groupby in operation.items():
                df[feature] = self.fill_na(df=df, col_name=feature, how=operation, groupby=groupby)
        return df

    @staticmethod
    def fill_na(*, df, col_name, how='mean', groupby=None):
        """Fill NA values with a chosen method
        """
        if groupby is not None:
            if how == 'FILL_NA_MEAN':
                return df.groupby(groupby)[col_name].transform(lambda x: x.fillna(x.mean()))
            elif how == 'FILL_NA_LAST_VALUE':
                return df.groupby(groupby)[col_name].transform(lambda x: x.fillna(x.shift(1)))
            elif how == 'FILL_NA_MEDIAN':
                return df.groupby(groupby)[col_name].transform(lambda x: x.fillna(x.median()))
        else:
            if how == 'FILL_NA_MEAN':
               return df[col_name].transform(lambda x: x.fillna(x.mean()))
            elif how == 'FILL_NA_LAST_VALUE':
                return df[col_name].transform(lambda x: x.fillna(x.shift(1)))
            elif how == 'FILL_NA_MEDIAN':
                return df[col_name].transform(lambda x: x.fillna(x.median()))
