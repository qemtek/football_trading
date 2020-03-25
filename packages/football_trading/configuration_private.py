import os
from packages.football_trading import project_dir

certs_dir = os.path.join(project_dir, 'certs')

# [DO NOT CHANGE] These are required to access the Betfair Exchange API.
betfair_credentials = {
    'betfairlightweight': {
        'username': 'qembet',
        'password': 'Double16!',
        'app_key': 'dNfmqo4rsDF6ygJl',
        'certs': certs_dir
    }
}