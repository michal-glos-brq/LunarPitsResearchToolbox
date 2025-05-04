import os
import logging

from astropy.time import Time

### System level confoguration
HDD_BASE_PATH = os.environ.get("WORKER_UTILITY_VOLUME", "/app/data")

## Run-through data files will be saved into HDD (SSD) or into RAM
SAVE_DATAFILES_TO_RAM = bool(os.environ.get("SAVE_DATAFILES_TO_RAM", True))

## Vanity
TQDM_NCOLS = 156

### Lunar configuration
LUNAR_RADIUS = 1737.4  # km

# Use more precise lunar model
# LUNAR_FRAME = "MOON_ME"
LUNAR_FRAME = "MOON_PA_DE440"

# If not interactive output, turn off TQDM
SUPRESS_TQDM = bool(os.environ.get("SUPRESS_TQDM", False))

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
