import logging
import os

from football_trading.src.utils.logging import get_console_handler
from football_trading.settings import PROJECTSPATH


# Configure logger for use in package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(get_console_handler())
logger.propagate = False


with open(os.path.join(PROJECTSPATH, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()