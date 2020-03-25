import betfairlightweight
import pandas as pd

from packages.football_trading.src.utils.betfair_tools import betfair_login
from packages.football_trading.src.utils.db import connect_to_db


def get_latest_fixtures():
    """Get the latest premier league fixtures from the Betfair Exchange API"""
    trading = betfair_login()
    trading.login()
    # Define what type of data to retrieve from the API
    market_projection = ["MARKET_START_TIME", "RUNNER_DESCRIPTION", "COMPETITION",
                         "EVENT", "MARKET_DESCRIPTION"]
    # Create a market filter
    event_filter = betfairlightweight.filters.market_filter(
                event_type_ids=[1],
                market_countries=['GB'],
                market_type_codes=['MATCH_ODDS'],
                text_query='English Premier League'
            )
    # Get market catalogue with event info
    market_catalogues = trading.betting.list_market_catalogue(
        filter=event_filter,
        market_projection=market_projection,
        max_results=200,
    )
    # Filter for premier league markets
    prem_markets = [market for market in market_catalogues
                    if market.competition.name == 'English Premier League']
    # Populate a DataFrame with the catalogue info
    df_odds = pd.DataFrame(columns=['market_id', 'home_team', 'away_team'])
    for market in prem_markets:
        names = market.event.name.split(' v ')
        # Get the selection IDs for home/draw/away
        home_id = [runner for runner in market.runners
                   if runner.runner_name == names[0]][0].selection_id
        away_id = [runner for runner in market.runners
                   if runner.runner_name == names[1]][0].selection_id
        # Add everything to a DataFrame
        df_odds = df_odds.append(
            pd.DataFrame({
                "market_id": market.market_id,
                'start_time': market.market_start_time,
                "home_team": names[0],
                "home_id": home_id,
                "away_team": names[1],
                "away_id": away_id},
                index=[len(df_odds)])
        )
    # Get market books (betting info) of the premier league markets
    market_books = trading.betting.list_market_book(
        market_ids=[market.market_id for market in prem_markets]
    )
    # Populate a DataFrame of the market book information
    df_mb = pd.DataFrame(columns=['market_id', 'selection_id', 'odds'])
    for book in market_books:
        for runner in book.runners:
            sid = {
                'market_id': book.market_id,
                'selection_id': runner.selection_id,
                'odds': runner.last_price_traded
                if runner.last_price_traded is not None else None
            }
            df_mb = df_mb.append(pd.DataFrame(sid, index=[len(df_mb)]))
    output_cols = ['market_id', 'market_start_time', 'home_team', 'home_id', 'away_id',
                   'away_team', 'home_odds', 'draw_odds', 'away_odds']
    # Check whether there are any markets open
    if len(df_odds) == 0 or len(df_mb) == 0:
        df_output = pd.DataFrame(columns=output_cols)
    else:
        # Merge the DataFrames containing market catalogue and market book info
        df_output = pd.merge(df_odds, df_mb,
                             left_on=['market_id', 'home_id'],
                             right_on=['market_id', 'selection_id'])
        df_output = pd.merge(df_output, df_mb[df_mb['selection_id'] == 58805],
                             on='market_id')
        df_output = pd.merge(df_output, df_mb,
                             left_on=['market_id', 'away_id'],
                             right_on=['market_id', 'selection_id'])
        df_output = df_output.drop(['selection_id', 'selection_id_x', 'selection_id_y'], axis=1)
        df_output.columns = ['away_id', 'away_team', 'home_id', 'home_team',
                             'market_id', 'market_start_time', 'home_odds', 'draw_odds', 'away_odds']
        df_output = df_output[['market_id', 'market_start_time', 'home_team', 'home_id', 'away_id',
                               'away_team', 'home_odds', 'draw_odds', 'away_odds']]
        # ToDo: Check how accurate last price traded is, look into getting odds the normal way
        #  if these odds arent similar to the betfair odds
    with connect_to_db() as conn:
        df_output.to_sql('bfex_latest_fixtures', conn, if_exists='replace')
