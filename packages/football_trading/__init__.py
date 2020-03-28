import os

from packages.football_trading.configuration import PROJECTSPATH

with open(os.path.join(PROJECTSPATH, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
