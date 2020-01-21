from flask import Flask, request, jsonify
import sys
import traceback

from src.models.XGBoostModel import XGBoostModel
from src.utils.api import get_upcoming_games
from update_tables import update_tables

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
    model = XGBoostModel()
    app.run(port=port, debug=False)
