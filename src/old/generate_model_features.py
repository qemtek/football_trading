from src.utils.db import connect_to_db, run_query
import pandas as pd
import numpy as np

window_length = 8

# Connect to database
conn = connect_to_db()

# Get all fixtures after game week 8, excluding the last game week
df = run_query("select t1.*, m_h.manager home_manager, m_h.start home_manager_start, "
                       "m_a.manager away_manager, m_a.start away_manager_start "
                       "from main_fixtures t1 "
                       "left join managers m_h "
                       "on t1.home_id = m_h.team_id "
                       "and (t1.date between m_h.start and date(m_h.end, '+1 day') "
                       "or t1.date > m_h.start and m_h.end is NULL) "
                       "left join managers m_a "
                       "on t1.away_id = m_a.team_id "
                       "and (t1.date between m_a.start and date(m_a.end, '+1 day') "
                       "or t1.date > m_a.start and m_a.end is NULL) "
                       "where t1.date > '2013-08-01'")

# Get additional features (time as manager) (manager age is logged to reduce scale)
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
df2 = run_query("select * from team_fixtures where date > '2013-08-01'")
df2['date'] = pd.to_datetime(df2['date'])
df2 = pd.merge(
    df2,
    df[['date', 'season', 'fixture_id', 'home_manager_age', 'away_manager_age',
        'home_manager_new', 'away_manager_new']],
    on=['date', 'season', 'fixture_id'])

# Filter out the first window_length and last game weeks from the data
df = df[(df['fixture_id'] > window_length*10) & (df['fixture_id'] < 370)]
# Filter out games that had red cards
df = df[(df['home_red_cards'] == 0) & (df['away_red_cards'] == 0)]

# Create categorical variables from match result
df2 = pd.get_dummies(df2, columns=['result'])

# Extract target
targets = df[['full_time_result']]
targets = pd.get_dummies(targets)


def calculate_win_streak(last_games):
    count = 0
    while(last_games.iloc[count] == 1):
        count += 1
    return count


def get_performance_vs_bookmaker(df):
    """Creates a score relating to how much the team beats
    (or doesn't beat) bookmaker expectations"""
    def calculate_score(row):
            return 1/row['b365_win_odds'] if row['result_W'] == 1 \
                else 1-1/row['b365_win_odds']
    df['perf_vs_bm'] = df.apply(lambda x: calculate_score(x), axis=1)
    return sum(df['perf_vs_bm'])


def get_home_away_advantage(df, type):
    df['goal_dif'] = df['goals_for'] - df['goals_against']
    home_data = df.loc[df['is_home'] == 1, :]
    away_data = df.loc[df['is_home'] == 0, :]
    home_goal_dif_sum = sum(home_data['goal_dif'])
    home_goal_dif_avg = np.mean(home_data['goal_dif'])
    away_goal_dif_sum = sum(away_data['goal_dif'])
    away_goal_dif_avg = np.mean(away_data['goal_dif'])
    home_advantage_sum = (0.1234 + home_goal_dif_sum)/(0.1234 + away_goal_dif_sum)
    home_advantage_avg = (0.1234 + home_goal_dif_avg) / (0.1234 + away_goal_dif_avg)
    return [home_advantage_sum, home_advantage_avg] if type == 'home' \
        else [1/home_advantage_sum, 1/home_advantage_avg]


def get_features(row, index, window_length=8, type='home'):
    team_name = row['home_team' if type == 'home' else 'away_team']
    fixture_id = row['fixture_id']
    season = row['season']
    # Filter for the team/season
    df_filtered = df2[(df2['team_name'] == team_name) &
       (df2['season'] == season) &
       (df2['fixture_id'] < fixture_id)]
    # Get the last 8 games
    df_filtered = df_filtered.sort_values('date').tail(window_length)
    # Create aggregated features
    df_output = pd.DataFrame()
    df_output.loc[index, 'avg_goals_for_'+type] = np.mean(df_filtered['goals_for'])
    df_output.loc[index, 'avg_goals_against_'+type] = np.mean(df_filtered['goals_against'])
    df_output.loc[index, 'sd_goals_for_'+type] = np.std(df_filtered['goals_for'])
    df_output.loc[index, 'sd_goals_against_'+type] = np.std(df_filtered['goals_against'])
    df_output.loc[index, 'avg_shots_for_'+type] = np.mean(df_filtered['shots_for'])
    df_output.loc[index, 'avg_shots_against_'+type] = np.mean(df_filtered['shots_against'])
    df_output.loc[index, 'sd_shots_for_'+type] = np.std(df_filtered['shots_for'])
    df_output.loc[index, 'sd_shots_against_'+type] = np.std(df_filtered['shots_against'])
    df_output.loc[index, 'avg_yellow_cards_'+type] = np.mean(df_filtered['yellow_cards'])
    df_output.loc[index, 'avg_red_cards_'+type] = np.mean(df_filtered['red_cards'])
    df_output.loc[index, 'b365_win_odds_'+type] = np.mean(df_filtered['b365_win_odds'])
    df_output.loc[index, 'avg_perf_vs_bm_'+type] = get_performance_vs_bookmaker(df_filtered)
    df_output.loc[index, 'manager_new_'+type] = row[type + '_manager_new']
    df_output.loc[index, 'manager_age_'+type] = row[type + '_manager_age']
    df_output.loc[index, 'win_rate_'+type] = np.mean(df_filtered['result_W'])
    df_output.loc[index, 'draw_rate_'+type] = np.mean(df_filtered['result_D'])
    df_output.loc[index, 'loss_rate_'+type] = np.mean(df_filtered['result_L'])
    ha_features = get_home_away_advantage(df_filtered, type)
    df_output.loc[index, 'home_advantage_sum_'+type] = ha_features[0]
    df_output.loc[index, 'home_advantage_avg_'+type] = ha_features[1]
    return df_output


num_features = 38
X = pd.DataFrame()
for i in range(len(df)):
    row = df.iloc[i, :]
    home_features = get_features(row, type='home', window_length=window_length, index=df.index[i])
    away_features = get_features(row, type='away', window_length=window_length, index=df.index[i])
    features = pd.concat([home_features, away_features], axis=1)
    X = X.append(features)

output = pd.concat([df[['fixture_id', 'date', 'home_team',
                        'home_id', 'away_team', 'away_id']],
                    X, targets], axis=1)

# Upload the data to a table in the database
run_query('DROP TABLE IF EXISTS model_features', return_data=False)
output.to_sql('model_features', conn)

# ToDo: Upload to model_features table
# ToDo: Add home/away stats (HA form)
# ToDo: Add whether the last game was an 'upset' from the b365 odds
