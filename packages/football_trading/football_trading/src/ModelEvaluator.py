import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from yellowbrick.classifier import ClassificationReport, DiscriminationThreshold, \
    ClassPredictionError, ConfusionMatrix, ROCAUC
from yellowbrick.regressor import ResidualsPlot, PredictionError

from football_trading.src.utils.general import time_function
from football_trading.src.utils.logging import get_logger
from football_trading.settings import plots_dir, data_dir, S3_BUCKET_NAME, LOCAL
from football_trading.src.utils.s3_tools import upload_to_s3

logger = get_logger()


class ModelEvaluator:
    """A class that stores functionality for evaluating model performance
    """
    def __init__(self, *, trained_model, model_id, training_data,
                 plots_dir, data_dir, is_classifier=True, local=True) -> None:
        self.trained_model = trained_model
        self.model_id = model_id
        LOCAL = local
        self.X_train = training_data.get('X_train')
        self.y_train = training_data.get('y_train')
        self.X_test = training_data.get('X_test')
        self.y_test = training_data.get('y_test')
        self.is_classifier = is_classifier
        self.plots_dir = plots_dir
        self.data_dir = data_dir
        # Generate plots, based on whether the model is a classifier or regressor
        if is_classifier:
            self.classification_report()
            #self.confusion_matrix()
            #self.discrimination_threshold()
            self.class_prediction_error()
        else:
            self.residuals_plot()
            self.prediction_error_plot()
        # Generate model-specific plots
        if trained_model.__class__.__name__ in ['XGBClassifier', 'XGBRegressor']:
            self.shap_summary()

    @time_function(logger=logger)
    def classification_report(self) -> None:
        """Show precision, recall and F1 score by class
        """
        visualizer = ClassificationReport(self.trained_model, cmap="YlGn", size=(600, 360))
        visualizer.fit(self.X_train, self.y_train)
        visualizer.score(self.X_test, self.y_test)
        save_dir = f"{self.plots_dir}/classification_report_{self.model_id}.png"
        visualizer.show(outpath=save_dir)
        if not LOCAL:
            upload_to_s3(save_dir, f'plots/classification_report_{self.model_id}.png', bucket=S3_BUCKET_NAME)
        plt.clf()

    @time_function(logger=logger)
    def discrimination_threshold(self) -> None:
        visualizer = DiscriminationThreshold(self.trained_model)
        visualizer.fit(self.X_test, self.y_test)  # Fit the data to the visualizer
        save_dir = f"{self.plots_dir}/discrimination_plot_{self.model_id}.png"
        visualizer.show(outpath=save_dir)
        if not LOCAL:
            upload_to_s3(save_dir, f'plots/discrimination_plot_{self.model_id}.png', bucket=S3_BUCKET_NAME)
        plt.clf()

    @time_function(logger=logger)
    def roc_curve(self, classes) -> None:
        visualizer = ROCAUC(self.trained_model, classes=classes)
        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        save_dir = f"{self.plots_dir}/roc_curve_{self.model_id}.png"
        visualizer.show(outpath=save_dir)
        if not LOCAL:
            upload_to_s3(save_dir, f'plots/roc_curve_{self.model_id}.png', bucket=S3_BUCKET_NAME)
        plt.clf()

    @time_function(logger=logger)
    def class_prediction_error(self) -> None:
        """Plot the support (number of training samples) for each class in the fitted classification
         model as a stacked bar chart. Each bar is segmented to show the proportion of predictions
         (including false negatives and false positives, like a Confusion Matrix) for each class.
         You can use a ClassPredictionError to visualize which classes your classifier is having
         a particularly difficult time with, and more importantly, what incorrect answers it is
         giving on a per-class basis.
         """
        visualizer = ClassPredictionError(self.trained_model)
        visualizer.fit(self.X_train, self.y_train)
        visualizer.score(self.X_test, self.y_test)
        save_dir = f"{self.plots_dir}/class_prediction_error_{self.model_id}.png"
        visualizer.show(outpath=save_dir)
        if not LOCAL:
            upload_to_s3(save_dir, f'plots/class_prediction_error_{self.model_id}.png', bucket=S3_BUCKET_NAME)
        plt.clf()

    @time_function(logger=logger)
    def confusion_matrix(self, class_name_dict=None) -> None:
        """Plot a confusion matrix
        """
        cm = ConfusionMatrix(
            self.trained_model,
            classes=list(class_name_dict.keys()),
            label_encoder=class_name_dict)
        cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)
        save_dir = f"{self.plots_dir}/confusion_matrix_{self.model_id}.png"
        cm.show(outpath=save_dir)
        if not LOCAL:
            upload_to_s3(save_dir, f'plots/confusion_matrix_{self.model_id}.png', bucket=S3_BUCKET_NAME)
        plt.clf()

    @time_function(logger=logger)
    def residuals_plot(self) -> None:
        """Plot the difference between the observed value of the target variable (y)
        and the predicted value (ŷ), i.e. the error of the prediction"""

        visualizer = ResidualsPlot(self.trained_model)
        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        save_dir = f"{self.plots_dir}/residuals_plot_{self.model_id}.png"
        visualizer.show(outpath=save_dir)
        if not LOCAL:
            upload_to_s3(save_dir, f'plots/residuals_plot_{self.model_id}.png', bucket=S3_BUCKET_NAME)
        plt.clf()

    @time_function(logger=logger)
    def prediction_error_plot(self) -> None:
        """Plot the actual targets from the dataset against the predicted values
        generated by our model. This allows us to see how much variance is in the model.
        """
        visualizer = PredictionError(self.trained_model)
        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        save_dir = f"{self.plots_dir}/prediction_error_plot_{self.model_id}.png"
        visualizer.show(outpath=save_dir)
        if not LOCAL:
            upload_to_s3(save_dir, f'plots/prediction_error_plot_{self.model_id}.png', bucket=S3_BUCKET_NAME)
        plt.clf()

    @time_function(logger=logger)
    def shap_summary(self, save_explainer=True):
        """Explain model predictions using SHAP
        """
        explainer = shap.TreeExplainer(self.trained_model)
        #X_sample = self.X_test.copy().sample(20000) if len(self.X_test) > 20000 else self.X_test
        X_sample = pd.concat([self.X_train, self.X_test], axis=0)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False, plot_size=(25, 10), class_names=self.trained_model.classes_)
        plt.savefig(f'{self.plots_dir}/summary_plot_{self.model_id}.png')
        plt.clf()
        if save_explainer:
            save_dir = f'{self.data_dir}/SHAP_explainer_{self.model_id}.joblib'
            with open(save_dir, 'wb') as f_out:
                joblib.dump(explainer, f_out)
            if not LOCAL:
                upload_to_s3(save_dir, f'plots/SHAP_explainer_{self.model_id}.jopblib', bucket=S3_BUCKET_NAME)
        return explainer, shap_values


if __name__ == '__main__':
    model = joblib.load('/Users/chriscollins/Documents/GitHub/football_trading/packages/football_trading/'
                        'football_trading/models/in_production/'
                        'match-predict-base_XGBClassifier_2020-03-19_9043841277632698831.joblib')
    training_data = joblib.load('/Users/chriscollins/Documents/GitHub/football_trading/packages/football_trading/'
                                'football_trading/data/training_data/'
                                'match-predict-base_XGBClassifier_2020-03-19_9043841277632698831.joblib')
    training_data['X_train'] = training_data['X_train'][model.model_features]
    training_data['X_test'] = training_data['X_test'][model.model_features]
    evaluator = ModelEvaluator(training_data=training_data, trained_model=model, data_dir=data_dir,
                               model_id='match-predict-base_XGBClassifier_2020-03-19_9043841277632698831',
                               is_classifier=True, plots_dir=plots_dir)
