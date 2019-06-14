from src.tools import connect_to_db, run_query
import scipy.stats


# Creates the poisson probabilities for goals_for and goals_against for each team, up to 3 goals.
def create_team_poisson_probabilities():

    # Connect to database
    conn, cursor = connect_to_db()

    # Extract data
    query = 'select fixture_id, team_name, date, season, is_home, goals_for, goals_against from team_fixtures'
    df = run_query(cursor, query)

    # Sort the data by season, team_name and date
    df = df.sort_values(['season', 'team_name', 'date'])

    # Calculate the moving average
    # NOTE: The moving average value uses the previous games data, not the goals_for and goals_against on that row.
    df['goals_for_mavg'] = df.groupby(['team_name', 'season'])['goals_for'].\
        transform(lambda x: x.rolling(8, 8).mean()).shift(1)
    df['goals_against_mavg'] = df.groupby(['team_name', 'season'])['goals_against'].\
        transform(lambda x: x.rolling(8, 8).mean()).shift(1)

    # Calculate the moving standard deviation
    #df['goals_for_sdavg'] = df.groupby(['team_name', 'season'])['goals_for'].\
    #    transform(lambda x: x.rolling(8, 8).std()).shift(1)
    #df['goals_against_sdavg'] = df.groupby(['team_name', 'season'])['goals_against'].\
    #    transform(lambda x: x.rolling(8, 8).std()).shift(1)

    # Get goals_for probabilities
    df['gf_prob_0'] = df['goals_for_mavg'].apply(lambda x: scipy.stats.distributions.poisson.pmf(0, x))
    df['gf_prob_1'] = df['goals_for_mavg'].apply(lambda x: scipy.stats.distributions.poisson.pmf(1, x))
    df['gf_prob_2'] = df['goals_for_mavg'].apply(lambda x: scipy.stats.distributions.poisson.pmf(2, x))
    df['gf_prob_3'] = df['goals_for_mavg'].apply(lambda x: scipy.stats.distributions.poisson.pmf(3, x))
    df['gf_prob_other'] = df.apply(lambda x: 1-(x['gf_prob_0'] + x['gf_prob_1'] +
                                                x['gf_prob_2'] + x['gf_prob_3']), axis=1)

    # Get goals_against probabilities
    df['ga_prob_0'] = df['goals_against_mavg'].apply(lambda x: scipy.stats.distributions.poisson.pmf(0, x))
    df['ga_prob_1'] = df['goals_against_mavg'].apply(lambda x: scipy.stats.distributions.poisson.pmf(1, x))
    df['ga_prob_2'] = df['goals_against_mavg'].apply(lambda x: scipy.stats.distributions.poisson.pmf(2, x))
    df['ga_prob_3'] = df['goals_against_mavg'].apply(lambda x: scipy.stats.distributions.poisson.pmf(3, x))
    df['ga_prob_other'] = df.apply(lambda x: 1 - (x['ga_prob_0'] + x['ga_prob_1'] +
                                                  x['ga_prob_2'] + x['ga_prob_3']), axis=1)

    # Create the DB table to store data
    cursor.execute("DROP TABLE IF EXISTS poisson_team_odds")
    cursor.execute("""CREATE TABLE poisson_team_odds (fixture_id INT, team_name TEXT, date DATE, season TEXT, 
    is_home INT, goals_for FLOAT, goals_against FLOAT, goals_for_mavg FLOAT, goals_against_mavg FLOAT, gf_prob_0 FLOAT, 
    gf_prob_1 FLOAT, gf_prob_2 FLOAT, gf_prob_3 FLOAT, ga_prob_0 FLOAT, ga_prob_1 FLOAT, ga_prob_2 FLOAT, 
    ga_prob_3 FLOAT)""")

    # Remove Nans (the first 8 games for each team)
    df = df.dropna()

    # Round floats to 2 decimal places
    float_columns = df.select_dtypes(include='float64').columns
    df[float_columns] = round(df.select_dtypes(include='float64'), 4)

    # Load data into the DB
    for row in df.iterrows():
        params = [row[1][0], str(row[1][1]), str(row[1][2]), str(row[1][3]), row[1][4], row[1][5], row[1][6],
                  row[1][7], row[1][8], row[1][9], row[1][10], row[1][11], row[1][12], row[1][13], row[1][14],
                  row[1][15], row[1][16]]
        cursor.execute('''insert into poisson_team_odds VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', params)

    conn.commit()


# Combines the poisson team_odds table so we can get the odds for different score combinations.
def combine_team_poisson_probabilities():

    # Connect to database
    conn, cursor = connect_to_db()

    # Extract data, join the poisson_team_odds table on itself to combine the home and away teams.
    query = '''
    select
        -- IDENTIFIERS
        fixture_id, date, season, home_team, away_team,
        -- SCORE PROBABILITIES
        0.5 * (gf_0_h + ga_0_a) * 0.5 * (gf_0_a + ga_0_h) p0_0,
        0.5 * (gf_1_h + ga_1_a) * 0.5 * (gf_0_a + ga_0_h) p1_0,
        0.5 * (gf_2_h + ga_2_a) * 0.5 * (gf_0_a + ga_0_h) p2_0,
        0.5 * (gf_3_h + ga_3_a) * 0.5 * (gf_0_a + ga_0_h) p3_0,
        0.5 * (gf_0_h + ga_0_a) * 0.5 * (gf_1_a + ga_1_h) p0_1,
        0.5 * (gf_1_h + ga_1_a) * 0.5 * (gf_1_a + ga_1_h) p1_1,
        0.5 * (gf_2_h + ga_2_a) * 0.5 * (gf_1_a + ga_1_h) p2_1,
        0.5 * (gf_3_h + ga_3_a) * 0.5 * (gf_1_a + ga_1_h) p3_1,
        0.5 * (gf_0_h + ga_0_a) * 0.5 * (gf_2_a + ga_2_h) p0_2,
        0.5 * (gf_1_h + ga_1_a) * 0.5 * (gf_2_a + ga_2_h) p1_2,
        0.5 * (gf_2_h + ga_2_a) * 0.5 * (gf_2_a + ga_2_h) p2_2,
        0.5 * (gf_3_h + ga_3_a) * 0.5 * (gf_2_a + ga_2_h) p3_2,
        0.5 * (gf_0_h + ga_0_a) * 0.5 * (gf_3_a + ga_3_h) p0_3,
        0.5 * (gf_1_h + ga_1_a) * 0.5 * (gf_3_a + ga_3_h) p1_3,
        0.5 * (gf_2_h + ga_2_a) * 0.5 * (gf_3_a + ga_3_h) p2_3,
        0.5 * (gf_3_h + ga_3_a) * 0.5 * (gf_3_a + ga_3_h) p3_3
    from
        (
        select 
            -- IDENTIFIERS
            t1.fixture_id, t1.date, t1.season, t1.team_name home_team, t2.team_name away_team,
            -- HOME TEAM PROBABILITIES
            t1.gf_prob_0 gf_0_h, t1.gf_prob_1 gf_1_h, t1.gf_prob_2 gf_2_h, t1.gf_prob_3 gf_3_h, 
            t1.ga_prob_0 ga_0_h, t1.ga_prob_1 ga_1_h, t1.ga_prob_2 ga_2_h, t1.ga_prob_3 ga_3_h, 
            -- AWAY TEAM PROBABILITIES
            t2.gf_prob_0 gf_0_a, t2.gf_prob_1 gf_1_a, t2.gf_prob_2 gf_2_a, t2.gf_prob_3 gf_3_a, 
            t2.ga_prob_0 ga_0_a, t2.ga_prob_1 ga_1_a, t2.ga_prob_2 ga_2_a, t2.ga_prob_3 ga_3_a
        from 
            (select * from poisson_team_odds where is_home = 1) t1 -- home_teams
        left join 
            (select * from poisson_team_odds where is_home = 0) t2 -- away_teams
        on t1.fixture_id = t2.fixture_id 
        and t1.season = t2.season 
        and t1.team_name is not t2.team_name
        )'''

    df = run_query(cursor, query)

    df['prob_sum'] = df.select_dtypes('float64').apply(lambda x: sum(x), axis=1)

    # Round floats to 2 decimal places
    float_columns = df.select_dtypes(include='float64').columns
    df[float_columns] = round(df.select_dtypes(include='float64'), 4)

    # Create the DB table to store data
    cursor.execute("DROP TABLE IF EXISTS poisson_match_probabilities")
    cursor.execute("""CREATE TABLE poisson_match_probabilities (fixture_id INT, date DATE, season TEXT, 
    home_team TEXT, away_team TEXT, p0_0 FLOAT, p1_0 FLOAT, p2_0 FLOAT, p3_0 FLOAT, p0_1 FLOAT, p1_1 FLOAT, 
    p2_1 FLOAT, p3_1 FLOAT, p0_2 FLOAT, p1_2 FLOAT, p2_2 FLOAT, p3_2 FLOAT, p0_3 FLOAT, p1_3 FLOAT, p2_3 FLOAT, 
    p3_3 FLOAT, prob_sum FLOAT)""")

    # Load data into the DB
    for row in df.iterrows():
        params = [row[1][0], str(row[1][1]), str(row[1][2]), str(row[1][3]), row[1][4], row[1][5], row[1][6],
                  row[1][7], row[1][8], row[1][9], row[1][10], row[1][11], row[1][12], row[1][13], row[1][14],
                  row[1][15], row[1][16], row[1][17], row[1][18], row[1][19], row[1][20], row[1][21]]
        cursor.execute('insert into poisson_match_probabilities VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', params)

    conn.commit()


# Run functions
create_team_poisson_probabilities()
combine_team_poisson_probabilities()



