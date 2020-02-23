from flask import Flask, request, jsonify
import sys
import traceback

from src.models.MatchResultXGBoostModel import MatchResultXGBoost
from src.utils.api import get_upcoming_games
from src.update_tables import update_tables
from src.utils.db import connect_to_db, run_query

from src.utils.base_model import get_logger

logger = get_logger(log_name='api')

# Your API definition
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        print(json_)
        prediction = model.predict(**json_)
        return jsonify(H=str(prediction.get('H')),
                       D=str(prediction.get('D')),
                       A=str(prediction.get('A')))
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/next_games', methods=['GET'])
def next_games():
    try:
        df = get_upcoming_games()
        return df.to_json()
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/historic_predictions', methods=['GET'])
def historic_predictions():
    try:
        predictions = model.get_historic_predictions()
        return predictions.to_json()
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/all_historic_predictions', methods=['GET'])
def all_historic_predictions():
    try:
        conn = connect_to_db()
        predictions = run_query('select * from historic_predictions')
        conn.close()
        return predictions.to_json()
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/latest_model_id', methods=['GET'])
def latest_model_id():
    try:
        return jsonify({'model_id': model.model_id})
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/update', methods=['POST'])
def update():
    try:
        update_tables()
    except:
        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 12345  # If you don't provide any port the port will be set to 12345
    model = MatchResultXGBoost(load_trained_model=True, problem_name='match_predict')
    app.run(port=port, debug=False)
