from flask import Flask, request, jsonify
from sklearn.externals import joblib
import sys
import traceback
import pandas as pd

from src.models.XGBoostModel import XGBoostModel

# Your API definition
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        print(json_)
        prediction = model.predict_matchup(pd.DataFrame(json_))
        return jsonify({'prediction': str(prediction)})
    except:
        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 12345  # If you don't provide any port the port will be set to 12345
    model = XGBoostModel(load_model=True)
    app.run(port=port, debug=True)