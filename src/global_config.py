import os

from astropy.time import Time

### System level confoguration
HDD_BASE_PATH = os.environ.get("WORKER_UTILITY_VOLUME", "/app/data")

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
