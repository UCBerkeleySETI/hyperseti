# Logging setup
# See https://logbook.readthedocs.io/en/stable/libraries.html
import sys
import logbook
from logbook import Logger, StreamHandler
StreamHandler(sys.stdout).push_application()

# Disable verbose hdf5plugin warning 
# https://github.com/silx-kit/hdf5plugin/issues/157
import logging
logging.getLogger('hdf5plugin').setLevel(logging.ERROR)


from .singletons import logger_name_list, logger_list

def update_levels(arg_group_level, arg_debug_name_list):
    for ix, logger_name in enumerate(logger_name_list):
        if logger_name in arg_debug_name_list:
            logger_list[ix].level = logbook.DEBUG
        else:
            logger_list[ix].level = arg_group_level


def get_logger(arg_name):
    logger = Logger(arg_name)
    logger.level = logbook.WARNING
    logger_name_list.append(arg_name)
    logger_list.append(logger)
    return logger
