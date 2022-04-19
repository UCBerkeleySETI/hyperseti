## Logging

Add a logger for each file:
```python
# Logging
from .log import logger_group, Logger
logger = Logger('hyperseti.utils')
logger_group.add_logger(logger)
```

To add a debug line in the logger:
```python
logger.debug(f"<name_of_func_here> Little message to explain: {val}")
```

To activate when debugging:
```python
from hyperseti.XX import logger
import logbook
logger.level = logbook.DEBUG
```

Or to activate all:
```python
from hyperseti import set_log_level
```

Where log level can be one of ('critical', 'error', 'warning', 'notice', 'info', 'debug')