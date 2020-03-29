from flask import Flask
from football_trading.dashboard import get_dashboard_app


def create_app(*, config_object) -> Flask:
    """Factory pattern to create a Flask api"""

    flask_app = Flask(__name__)
    # Import blueprints
    from api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)
    # Get configuration.py from the config object passed by the user
    flask_app.config.from_object(config_object)
    # Run the dash dashboard and add it to the flask app
    get_dashboard_app(server=flask_app)
    return flask_app


if __name__ == '__main__':
    app = create_app()
