from src.models.XGBoostModel import XGBoostModel

# Create an instance of the XGBoost model, which automatically
# trains on the latest data (if you dont specify to load a model)
model = XGBoostModel(load_latest_model=False)