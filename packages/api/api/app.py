from flask import Flask, request, jsonify

from football_trading import __version__ as _version
from api import __version__ as api_version
from football_trading.predict import make_predictions
from football_trading.train import train_new_model
from football_trading.dashboard import get_dashboard_app
from settings import config
from api.config import get_logger

_logger = get_logger(logger_name=__name__)


def get_api():
    # Create API object
    prediction_app = Flask(__name__)
    # Import config
    prediction_app.config.from_object(config)

    @prediction_app.route('/health', methods=['GET'])
    def health():
        if request.method == 'GET':
            _logger.info('Health status: OK')
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
    def update_dashboard():
        if request.method == 'GET':
            _logger.info(f'Starting/refreshing model performance dashboard')
            # Run the dash dashboard and add it to the flask app
            get_dashboard_app(server=prediction_app)

    @prediction_app.route('/version', methods=['GET'])
    def get_version():
        return jsonify({"API version": api_version,
                        "Model version": _version})

    return prediction_app
