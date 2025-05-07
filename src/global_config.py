import os
import logging

from tqdm import tqdm
from astropy.time import Time

### System level confoguration
HDD_BASE_PATH = os.environ.get("WORKER_UTILITY_VOLUME", "/app/data")

## Run-through data files will be saved into HDD (SSD) or into RAM
SAVE_DATAFILES_TO_RAM = bool(os.environ.get("SAVE_DATAFILES_TO_RAM", False))

## Vanity
TQDM_NCOLS = 156

### Lunar configuration
LUNAR_RADIUS = 1737.4  # km

# Use more precise lunar model
# LUNAR_FRAME = "MOON_ME"
LUNAR_FRAME = "MOON_PA_DE440"

# If not interactive output, turn off TQDM
SUPRESS_TQDM = bool(os.environ.get("SUPRESS_TQDM", False))
# On some places, we print the traceback of unexpected but caught exceptions, this will supress it, if set to True
SUPRESS_TRACEBACKS = False

# Max precision of stropy time
SPICE_DECIMAL_PRECISION = 9
Time.precision = SPICE_DECIMAL_PRECISION

MASTER_IP = os.getenv("MASTER_IP", "host.docker.internal")

# DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
if LOG_LEVEL not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    raise ValueError(f"Invalid log level: {LOG_LEVEL}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")

LOG_LEVEL = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}[LOG_LEVEL]



class TqdmLoggingHandler(logging.Handler):
    """A handler that routes all logs through tqdm.write()."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

setup_flag = False

def setup_logging():
    global setup_flag
    if setup_flag:
        return
    setup_flag = True

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any handlers configured earlier
    for h in root.handlers[:]:
        root.removeHandler(h)

    if not SUPRESS_TQDM:
        handler = TqdmLoggingHandler()
    else:
        handler = logging.StreamHandler()

    fmt = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)