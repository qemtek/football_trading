import os

from packages.football_trading.configuration import project_dir

with open(os.path.join(project_dir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
