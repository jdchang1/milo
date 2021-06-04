"""
Logger singleton wrapper
Default logger folder is `os.path.join(__file__, '..', '..', 'logs')`
"""
import logging
import logging.handlers
import os
import sys


__all__ = ['init_logger']


def init_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_level = logging.DEBUG
    log_format = '%(message)s'

    logger = logging.getLogger(log_dir)
    logger.setLevel(log_level)
    path = os.path.join(log_dir, 'main.log')

    # file handler (log file)
    log_handler = logging.handlers.RotatingFileHandler(filename=path)
    log_handler.setLevel(log_level)
    log_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(log_handler)

    # stream handler (default sys.stderr)
    log_handler = logging.StreamHandler()
    log_handler.setLevel(log_level)
    log_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(log_handler)

    return logger
