# Logging setup
# See https://logbook.readthedocs.io/en/stable/libraries.html
import sys
import logbook
from logbook import Logger, StreamHandler

StreamHandler(sys.stdout).push_application()
logger_group = logbook.LoggerGroup()
logger_group.level = logbook.INFO
debug_list = []

def init_logging(arg_group_level, arg_debug_list):
    logger_group.level = arg_group_level
    for logger_name in arg_debug_list:
        debug_list.append(logger_name)
    
def get_logger(arg_name):
    logger = Logger(arg_name)
    logger_group.add_logger(logger)
    if arg_name in debug_list:
        logger.level = logbook.DEBUG
    return logger
