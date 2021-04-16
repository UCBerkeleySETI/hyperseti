# Logging setup
# See https://logbook.readthedocs.io/en/stable/libraries.html
import sys
import logbook
from logbook import Logger, StreamHandler
StreamHandler(sys.stdout).push_application()

logger_group = logbook.LoggerGroup()
logger_group.level = logbook.WARNING