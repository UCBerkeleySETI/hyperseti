from .hyperseti import run_pipeline, find_et
#from .dedoppler import dedoppler, apply_boxcar, normalize
from .hits import hitsearch
from .version import HYPERSETI_VERSION
from .log import logger_group
from .dedoppler import dedoppler
from .normalize import normalize

__version__ = HYPERSETI_VERSION