from configuration_private import betfair_credentials
import betfairlightweight


def betfair_login():
    """Log into the Betfair Exchange API using credentials stored in configuration.py"""
    usr = betfair_credentials["betfairlightweight"]["username"]
    pw = betfair_credentials["betfairlightweight"]["password"]
    app_key = betfair_credentials["betfairlightweight"]["app_key"]
    certs = betfair_credentials["betfairlightweight"]["certs"]
    trading = betfairlightweight.APIClient(usr, pw, app_key=app_key, certs=certs)
    return trading
