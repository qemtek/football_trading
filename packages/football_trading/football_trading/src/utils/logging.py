import logging
import sys

from logging.handlers import TimedRotatingFileHandler

from settings import FORMATTER, LOG_FILE


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
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
