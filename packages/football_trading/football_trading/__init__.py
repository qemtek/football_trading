from football_trading.settings import PROJECTSPATH

version_dir = 'football_trading/VERSION'
with open(version_dir, 'r') as version_file:
    __version__ = version_file.read().strip()
