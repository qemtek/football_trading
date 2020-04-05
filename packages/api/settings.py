import logging

from utils.configuration import get_attribute
from api.config import ProductionConfig, DevelopmentConfig, TestingConfig

SERVER_ADDRESS = get_attribute('SERVER_ADDRESS')
SERVER_PORT = get_attribute('SERVER_PORT')
PROJECTSPATH = get_attribute('PROJECTSPATH')

MODE = get_attribute('MODE', accepts=['prod', 'dev', 'test'], fail_if_not_found=True)
if MODE == 'prod':
    config = ProductionConfig
elif MODE == 'dev':
    config = DevelopmentConfig
elif MODE == 'test':
    config = TestingConfig

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(funcName)s:%(lineno)d - %(message)s")
LOG_FILE = f"{PROJECTSPATH}/logs/model.log"
