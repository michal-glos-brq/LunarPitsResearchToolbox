"""
============================================================
Global configuration module
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

import os
import sys
import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
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

_LOG_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
if _LOG_NAME not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    raise ValueError(f"Invalid LOG_LEVEL { _LOG_NAME }")
LOG_LEVEL = getattr(logging, _LOG_NAME)


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
        # wrap so that any calls inside a tqdm context stay in order
        logging_redirect_tqdm()
        handler = TqdmLoggingHandler(LOG_LEVEL)
    else:
        handler = logging.StreamHandler(stream=sys.stderr)

    fmt = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
