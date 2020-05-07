import betfairlightweight

from football_trading.settings import BFEX_USER, BFEX_PASSWORD, BFEX_APP_KEY, BFEX_CERTS_PATH


def connect_to_betfair():
    """Log into the Betfair Exchange API using credentials stored in the environment/logging.py"""
    trading = betfairlightweight.APIClient(
        username=BFEX_USER, password=BFEX_PASSWORD, app_key=BFEX_APP_KEY, certs=BFEX_CERTS_PATH)
    return trading
