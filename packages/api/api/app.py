from flask import Flask


def create_app(*, config_object) -> Flask:
    """Factory pattern to create a Flask api"""

    flask_app = Flask(__name__)
    # Import blueprints
    from api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)
    # Get configuration.py from the config object passed by the user
    flask_app.config.from_object(config_object)
    return flask_app


if __name__ == '__main__':
    app = create_app()
