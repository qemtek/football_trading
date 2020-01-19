# Models for football prediction

This projects has been created to design and test different models for the prediction of football games. The aim of these models is to provide pre-match probabilities to a real-time odds trading model (private project) which will operate on the Betfair Exchange API.

### Current models being developed

- XGBoost - (90% Complete)

- LSTM Recurrent Neural Network (30% Complete)

To DO

- Wrap the model in an API and test its ability to serve predictions

- Create a dashboard to monitor model performance and post predictions

- Host the dashboard on a server, seperated from the model API

- Each week, download the fixtures for the upcoming game week and serve them to the model API.

- Load the previous years player-level data into this DB and experiment with player-level features.