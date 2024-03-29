# Logging setup
# See https://logbook.readthedocs.io/en/stable/libraries.html
# Levels are:
#   critical - for errors that lead to termination
#   error - for errors that occur, but are handled
#   warning - for exceptional circumstances that might not be errors
#   notice - for non-error messages you usually want to see
#   info - for messages you usually don't want to see
#   debug - for debug messages

import sys
import logbook
from logbook import Logger, StreamHandler

# Start logging to screen (stdout)
log_to_screen = StreamHandler(sys.stdout, bubble=True)
log_to_screen.format_string = '[{record.channel}] {record.message}'
log_to_screen.push_application()

# Disable verbose hdf5plugin warning 
# https://github.com/silx-kit/hdf5plugin/issues/157
import logging
logging.getLogger('hdf5plugin').setLevel(logging.ERROR)

from .singletons import logger_name_list, logger_list

LOG_LEVELS = {
    'critical': logbook.CRITICAL,
    'error': logbook.ERROR,
    'warning': logbook.WARNING,
    'notice': logbook.NOTICE,
    'info': logbook.INFO,
    'debug': logbook.DEBUG
}

def set_log_level(level: str, debug: list=[]):
    """ Set global logging level for hyperseti calls
    
    Args:
        level (str): one of (critical, error, warning, notice, info, debug)
        debug (list of strs): List of modules to set to DEBUG mode.
    """
    update_levels(level, debug)

def update_levels(arg_group_level: str, arg_debug_name_list: list=[]):
    """ Update logging levels 
    
    Args:
        arg_group_level (str or logbook LEVEL): Level to set loggers to
        arg_debug_name_list (list): List of logger names to set to debug status
    
    Notes:
        This is called after all the get_logger calls 
        (loading) are done and while the program is being executed. 
        For each element of the logger_name_list & logger_list (singletons).

    """

    if arg_group_level in LOG_LEVELS.keys():
        arg_group_level = LOG_LEVELS[arg_group_level]

    for ix, logger_name in enumerate(logger_name_list):
        if logger_name in arg_debug_name_list:
            logger_list[ix].level = logbook.DEBUG
        else:
            logger_list[ix].level = arg_group_level


def get_logger(arg_name: str, level: str=None) -> Logger:
    """ Called when the signal processing functions are being 
    loaded by Python. Each call returns a logger object whose level will be 
    updated later. Successive calls build up 2 singleton lists: logger_list 
    and logger_name_list.

    Args:
        arg_name (str): Logger name, e.g. hyperseti.utils
        level (str or None): Sets logger level to one of (critical, error, warning, notice, info, debug)
    """
    if arg_name in logger_name_list:
        logger = logger_list[logger_name_list.index(arg_name)]
    else:
        logger = Logger(arg_name)
        logger.level = logbook.WARNING
        logger_name_list.append(arg_name)
        logger_list.append(logger)

    if level is not None:
        logger.level = LOG_LEVELS.get(level, level)   
    return logger
