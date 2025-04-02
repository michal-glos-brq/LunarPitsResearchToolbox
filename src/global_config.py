import os

### System level confoguration
HDD_BASE_PATH = os.environ.get("UTILITY_VOLUME", "/media/mglos/HDD_8TB4/TMP")

## Vanity
TQDM_NCOLS = 156

### Lunar configuration
LUNAR_RADIUS = 1737.4  # km

# Use more precise lunar model
#LUNAR_FRAME = "MOON_ME"
LUNAR_FRAME = "MOON_PA_DE440"

# If not interactive output, turn off TQDM
SUPRESS_TQDM = False
