from src.models.XGBoostModel import XGBoostModel
import logging

logger = logging.getLogger('XGBoostModel')

# Create an instance of the XGBoost model, which automatically
# trains on the latest data (if you dont specify to load a model)
model = XGBoostModel(load_model=True)
prediction = model.predict_matchup(1, 8, '2020-01-10', "19/20")

# ToDo: Load the API in one thread

# ToDo: Load the dashboard in another thread
