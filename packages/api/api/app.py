from flask import Flask, request, jsonify

from football_trading import __version__ as _version
from api import __version__ as api_version
from football_trading.predict import make_predictions
from football_trading.train import train_new_model
from football_trading.dashboard import get_dashboard_app
from api.settings import config
from api.utils.logging import get_logger

_logger = get_logger(logger_name=__name__)


def get_api(input_config=None):
    # Create API object
    prediction_app = Flask(__name__)
    # Import config
    prediction_app.config.from_object(input_config if input_config is not None else config)

    @prediction_app.route('/health', methods=['GET'])
    def health():
        if request.method == 'GET':
            _logger.info('Health status: OK')
            return 'ok'

    @prediction_app.route('/predict', methods=['GET'])
    def predict():
        if request.method == 'GET':
            _logger.info(f'Making new predictions')
            try:
                make_predictions()
                return 200
            except:
                return 500

    @prediction_app.route('/train', methods=['GET'])
    def train():
        if request.method == 'GET':
            _logger.info(f'Training new model')
            try:
                train_new_model()
                return 200
            except:
                return 500

    # @prediction_app.route('/update_dashboard', methods=['GET'])
    # def update_dashboard():
    #     if request.method == 'GET':
    #         _logger.info(f'Starting/refreshing model performance dashboard')
    #         # Run the dash dashboard and add it to the flask app
    #         get_dashboard_app(server=prediction_app)
    #         return 'successful'

    @prediction_app.route('/version', methods=['GET'])
    def get_version():
        return jsonify({"API version": api_version,
                        "Model version": _version})

    return prediction_app
