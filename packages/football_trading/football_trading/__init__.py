import logging

from football_trading.settings import PROJECTSPATH
from football_trading.src.utils.logging import get_console_handler

# Configure logger for use in package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(get_console_handler())
logger.propagate = False

version_dir = f"{PROJECTSPATH}/football_trading/VERSION"
logger.info(f"Getting version from {version_dir}")
with open(version_dir, 'r') as version_file:
    __version__ = version_file.read().strip()
