from src.tools import connect_to_db, run_query
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import tensorflow
from keras import regularizers
from keras.activations import tanh, sigmoid
from keras.layers import Dropout, Dense, Flatten
from keras.models import Sequential
from keras.layers import LSTM, Masking
from keras.optimizers import adam
from keras.callbacks import History
import keras.backend as K
import tensorflow as tf
import talos as ta
from sklearn.model_selection import KFold
from sklearn.utils import multiclass
from sklearn.metrics import accuracy_score
import datetime as dt
from sklearn.utils.class_weight import compute_class_weight
import math

# Connect to database
conn, cursor = connect_to_db()

# Get all fixtures after game week 8, excluding the last game week
df = run_query(cursor, "select t1.*, m_h.manager home_manager, m_h.start home_manager_start, "
                       "m_a.manager away_manager, m_a.start away_manager_start "
                       "from main_fixtures t1 "
                       "left join managers m_h "
                       "on t1.home_id = m_h.team_id "
                       "and (t1.date between m_h.start and date(m_h.end, '+1 day') or t1.date > m_h.start and m_h.end is NULL) "
                       "left join managers m_a "
                       "on t1.away_id = m_a.team_id "
                       "and (t1.date between m_a.start and date(m_a.end, '+1 day') or t1.date > m_a.start and m_a.end is NULL) "
                       "where t1.date > '2013-08-01'")

# Get additional features (time as manager)
df['date'] = pd.to_datetime(df['date'])
df['home_manager_start'] = pd.to_datetime(df['home_manager_start'])
df['home_manager_age'] = df.apply(
    lambda x: np.log10(round((x['date'] - x['home_manager_start']).days)), axis=1)
df['away_manager_start'] = pd.to_datetime(df['away_manager_start'])
df['away_manager_age'] = df.apply(
    lambda x: np.log10(round((x['date'] - x['away_manager_start']).days)), axis=1)
df['home_manager_new'] = df['home_manager_age'].apply(lambda x: 1 if x <= 70 else 0)
df['away_manager_new'] = df['away_manager_age'].apply(lambda x: 1 if x <= 70 else 0)

# Get team stats
df2 = run_query(cursor, "select * from team_fixtures where date > '2013-08-01'")
df2['date'] = pd.to_datetime(df2['date'])
df2 = pd.merge(df2, df[['date', 'season', 'fixture_id',
                       'home_manager_age', 'away_manager_age', 'home_manager_new',
                       'away_manager_new']], on=['date', 'season', 'fixture_id'])
# Get the managers age, regardless of home or away team
df2['manager_age'] = df2.apply(
    lambda x: x['home_manager_age']
    if x['is_home'] == 1 else x['away_manager_age'], axis=1)
df2['manager_new'] = df2.apply(
    lambda x: x['home_manager_new']
    if x['is_home'] == 1 else x['away_manager_new'], axis=1)
df2 = df2.drop(['home_manager_age', 'home_manager_new',
                'away_manager_age', 'away_manager_new'], axis=1)

# Filter out the first 8 and last game weeks from the data
df = df[(df['fixture_id'] > 80) & (df['fixture_id'] < 370)]
# Filter out games that had red cards
df = df[(df['home_red_cards'] == 0) & (df['away_red_cards'] == 0)]

# Filter out the first 8 game weeks, and the last game week of the season
df2['game_number'] = df2.sort_values('date'). \
    groupby(['team_name', 'season'])['fixture_id']. \
    rank(method="first", ascending=True)

features = ['goals_for', 'goals_against', 'goal_difference', 'yellow_cards', 'red_cards',
            'shots_for', 'shots_against', 'shot_difference', 'b365_win_odds', 'is_home',
            'manager_new', 'manager_new']

window_length = 8
num_games = len(df)
additional_features = 0
num_features = (len(features) + additional_features) * 2
df2 = pd.get_dummies(df2, columns=['result'])

# def two_class_target(result):
#     return 1 if result == 'H' else 0
#
# df['full_time_result'] = df['full_time_result'].apply(lambda x: two_class_target(x))

# Extract target
targets = df[['full_time_result']]
targets = pd.get_dummies(targets)

# Preprocessing
scaler_ML = QuantileTransformer(output_distribution='normal')
df2[features] = scaler_ML.fit_transform(df2[features])


#features = features + ['win_streak']

def calculate_win_streak(last_games):
    count = 0
    while(last_games.iloc[count] == 1):
        count += 1
    return count


# Create the proper shape of the data (rows * features * window_length)
def get_last_8_games(row, window_length=8, type='home'):
    team_name = row['home_team' if type == 'home' else 'away_team']
    fixture_id = row['fixture_id']
    season = row['season']
    # Filter for the team/season
    df_filtered = df2[(df2['team_name'] == team_name) &
       (df2['season'] == season) &
       (df2['fixture_id'] < fixture_id)]



    # ToDo: Add difference statistics
    # ToDo: Only include the manager stats for that team

    # Get the last 8 games
    df_filtered = df_filtered.sort_values('date', ascending=True).tail(window_length)

    df_filtered = np.zeros((window_length, int(num_features/2))) \
        if len(df_filtered) != 8 else np.array(df_filtered[features])
    return df_filtered


X = np.zeros((num_games, window_length, num_features))
for i in range(len(df)):
    X[i, :, 0:int(num_features/2)] = get_last_8_games(df.iloc[i, :], type='home')
    X[i, :, int(num_features/2):num_features] = get_last_8_games(df.iloc[i, :], type='away')

# Filter out any arrays with zeros
to_delete = []
for i in range(len(df)):
    if X[i, 0, 0] == 0 or X[i, 0, num_features-1] == 0:
        to_delete.append(i)
X = np.delete(X, obj=to_delete, axis=0)
num_games = len(X)
targets = np.delete(np.array(targets), to_delete, axis=0)
#df = df.iloc[~df.index.isin(to_delete), :]


# Split data into test/train
mask=np.random.rand(num_games) < 0.8
X_test = X[~mask, :, :]
X_train = X[mask, :, :]
y_test = targets[~mask]
y_train = targets[mask]

mask_value = -500.1234

# class_weights = compute_class_weight(
#     'balanced', np.unique(df.iloc[mask, :]['full_time_result']),
#     df.iloc[mask, :]['full_time_result'])


# def get_class_weights(row):
#     if row[0] == 1:
#         return class_weights[0]
#     elif row[1] == 1:
#         return class_weights[1]
#     elif row[2] == 1:
#         return class_weights[2]
#     else:
#         return -1

# define general model for hyperparameter tuning
def lstm_model(x_train, y_train, x_val, y_val, params):

    # weights = np.apply_along_axis(get_class_weights, 1, y_train)
    history = History()
    model = Sequential()
    # model.add(Flatten())
    model.add(Masking(mask_value=mask_value))
    model.add(Dense(params['dense_neurons'],
                    activation=params['non_lstm_act'],
                    kernel_regularizer=regularizers.l1(params['l1_reg']),
                    input_shape=(x_train.shape[1], x_train.shape[2],)
                   ))
    model.add(Dropout(rate=params['dropout']))
    model.add(LSTM(params['lstm_hspace'],
                   kernel_regularizer=regularizers.l2(params['l2_reg']),
                   return_sequences=True,
                   activation='tanh',
                   ))
    model.add(LSTM(params['lstm_hspace'],
                   kernel_regularizer=regularizers.l2(params['l2_reg']),
                   return_sequences=False,
                   activation='tanh',
                   ))
    # model.add(Dense(round(params['dense_neurons']/2),
    #                 activation=params['non_lstm_act'],
    #                 kernel_regularizer=regularizers.l2(params['l1_reg'])))

    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam(lr=params['learning_rate']),
                  metrics=['accuracy'])

    # ToDo: Weight the loss by the potential return
    # ToDo: Try weighting by home/draw/away
    # ToDo: Add a callback to stop on a certain level of accuracy

    out = model.fit(x_train, y_train,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  verbose=1,
                  validation_data=(x_val, y_val),
                  steps_per_epoch=None,
                  # sample_weight=weights,
                  callbacks=[ta.utils.ExperimentLogCallback(experiment_name, params)])

    return out, model

experiment_name = 'test1'

p = {'dense_neurons': [32], 'lstm_hspace': [4],
     'batch_size': [1], 'dropout': [0.2],
     'l1_reg': [0.01], 'l2_reg': [0.0100], 'epochs': [10],
     'learning_rate': [0.001], 'non_lstm_act': [tanh]}
scan_object = ta.Scan(X_train, y_train, model=lstm_model,
                      params=p, experiment_name=experiment_name)
r = ta.Reporting(scan_object)

print(r.data[['val_accuracy', 'epochs', 'batch_size',
              'learning_rate', 'dense_neurons', "lstm_hspace"]].sort_values(
    'val_accuracy', ascending=False))

# number of iterations in k-fold validation
folds = 5
# talos calls using the best model
p = ta.Predict(scan_object, task='multi_class')
e = ta.Evaluate(scan_object)
accuracy_scores = e.evaluate(
    X_test, y_test, folds=folds, task='multi_label', metric='val_accuracy')
predictions = p.predict(X_test, metric='val_accuracy')
print('F1: ', np.mean(accuracy_scores))


def get_actual_class(row):
    if row.loc[0] == 1:
        return 0
    elif row.loc[1] == 1:
        return 1
    elif row.loc[2] == 1:
        return 2


y_actual = pd.DataFrame(y_test).apply(lambda x: get_actual_class(x), axis=1)
class_predictions = pd.DataFrame(p.predict_classes(X_test, metric='val_accuracy'))
class_predictions.columns = ['prediction']
class_predictions['actual'] = y_actual
class_predictions.groupby('prediction').count()
accuracy_score(class_predictions['actual'], class_predictions['prediction'])

class_predictions.groupby('actual').count()/len(class_predictions)

# ToDo: Evaluate the model on the expected return
# ToDo: Add opponent difficulty (avg goal difference)
# ToDo: Add days since last game and days until next game
# ToDo: Can we scrape the lineup?

y_actual_train = pd.DataFrame(y_train).apply(lambda x: get_actual_class(x), axis=1)
class_predictions_train = pd.DataFrame(p.predict_classes(X_train, metric='val_accuracy'))
class_predictions_train.columns = ['prediction']
class_predictions_train['actual'] = y_actual_train
class_predictions_train.groupby('prediction').count()
accuracy_score(class_predictions_train['actual'], class_predictions_train['prediction'])