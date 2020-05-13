from football_trading.settings import PROJECTSPATH

version_dir = f'{PROJECTSPATH}/VERSION'
with open(version_dir, 'r') as version_file:
    __version__ = version_file.read().strip()
