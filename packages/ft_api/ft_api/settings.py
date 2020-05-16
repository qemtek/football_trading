import logging
import os

from utils.config import get_attribute

SERVER_ADDRESS = get_attribute('SERVER_ADDRESS')
SERVER_PORT = get_attribute('SERVER_PORT')
PROJECTSPATH = get_attribute('PROJECTSPATH')

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")

# Create log directory
LOG_DIR = f"{PROJECTSPATH}/logs"
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FILE = f"{LOG_DIR}/ft_api.log"

# Create upload folder
UPLOAD_FOLDER = f"{PROJECTSPATH}/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
class Config:
    DEBUG = False
    TESTING = False
    DEVELOPMENT = False
    CSRF_ENABLED = True
    #SECRET_KEY = 'this-really-needs-to-be-changed'
    UPLOAD_FOLDER = UPLOAD_FOLDER
    SERVER_ADDRESS = get_attribute('SERVER_ADDRESS')
    SERVER_PORT = get_attribute('SERVER_PORT')


class ProductionConfig(Config):
    DEBUG = False


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True


MODE = get_attribute('MODE', accepts=['prod', 'dev', 'test'], fail_if_not_found=True)
if MODE == 'prod':
    config = ProductionConfig
elif MODE == 'dev':
    config = DevelopmentConfig
elif MODE == 'test':
    config = TestingConfig

