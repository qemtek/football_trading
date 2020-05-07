import sys
import logging
from logging.handlers import TimedRotatingFileHandler

from ft_api import configuration


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(configuration.FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(configuration.LOG_FILE, when='midnight')
    file_handler.setFormatter(configuration.FORMATTER)
    file_handler.setLevel(logging.INFO)
    return file_handler


def get_logger(logger_name=__name__):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger
