from flask import Blueprint, request

from football_trading.predict import make_predictions
from football_trading.train import train_new_model
from football_trading.dashboard import get_dashboard_app

from api.config import get_logger

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        return 'ok'


@prediction_app.route('/v1/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        _logger.info(f'Making new predictions')
        make_predictions()


@prediction_app.route('/v1/train', methods=['POST'])
def train():
    if request.method == 'POST':
        _logger.info(f'Training new model')
        train_new_model()


@prediction_app.route('/v1/update_dashboard', methods=['GET'])
def dashboard():
    if request.method == 'GET':
        _logger.info(f'Starting/refreshing model performance dashboard')
        app = get_dashboard_app(server=prediction_app)
        return app.index()
