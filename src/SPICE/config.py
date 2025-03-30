import os
import urllib
import numpy as np
from functools import partial
from astropy.time import TimeDelta

from src.global_config import HDD_BASE_PATH


##########################################################################################
#####                                 SPICE UTILS                                    #####
##########################################################################################


### Limit concurrent kernel downloads
MAX_KERNEL_DOWNLOADS = 12
MAX_LOADED_DYNAMIC_KERNELS = 3
KEEP_DYNAMIC_KERNELS = False

SPICE_CHUNK_SIZE = 8192
SPICE_READ_TIMEOUT = 15
SPICE_TOTAL_TIMEOUT = 60


### Some SpiceyPy constants
SPICE_FOLDER = os.path.join(HDD_BASE_PATH, "SPICE")
SPICE_PERSIST = True

KERNEL_TIME_KEYS = {"filename_key": "^SPICE_KERNEL", "time_start_key": "START_TIME", "time_stop_key": "STOP_TIME"}
SECOND_TIMEDELTA = TimeDelta(1, format="sec")
MAX_RETRIES = 250
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}



##########################################################################################
#####                                 SPICE FRAMES                                   #####
##########################################################################################
MOON_STR_ID = "MOON"
LRO_STR_ID = "LUNAR RECONNAISSANCE ORBITER"
ABBERRATION_CORRECTION = "CN+S"

LRO_DIVINER_FRAME_STR_ID = "LRO_DLRE"
LRO_LOLA_FRAME_STR_ID = "LRO_LOLA"
LRO_MINIRF_FRAME_STR_ID = "LRO_MINIRF"


##########################################################################################
#####                           SPICE KERNEL LOCATIONS                               #####
##########################################################################################
LRO_SPICE_KERNEL_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/"
GRAIL_SPICE_KERNEL_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/"
SELENE_SPICE_KERNEL_BASE_URL = "https://darts.isas.jaxa.jp/pub/spice/SELENE/kernels_ORG/"
CHANDRAYAAN_SPICE_KERNEL_BASE_URL = ""
GENERIC_SPICE_KERNEL_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"

## And helper function
grail_path = partial(os.path.join, SPICE_FOLDER, "grail")
lro_path = partial(os.path.join, SPICE_FOLDER, "lro")
selene_path = partial(os.path.join, SPICE_FOLDER, "selene")
chandrayaan_path = partial(os.path.join, SPICE_FOLDER, "chandrayaan")
root_path = partial(os.path.join, SPICE_FOLDER)

grail_url = partial(urllib.parse.urljoin, GRAIL_SPICE_KERNEL_BASE_URL)
lro_url = partial(urllib.parse.urljoin, LRO_SPICE_KERNEL_BASE_URL)
selene_url = partial(urllib.parse.urljoin, SELENE_SPICE_KERNEL_BASE_URL)
chandrayaan_url = partial(urllib.parse.urljoin, CHANDRAYAAN_SPICE_KERNEL_BASE_URL)
generic_url = partial(urllib.parse.urljoin, GENERIC_SPICE_KERNEL_BASE_URL)


##########################################################################################
#####                                     DSK                                        #####
##########################################################################################
LUNAR_TIF_DATA_URL = "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

BUNNY_PASSWORD = "a8b12405-7fc0-422e-845602481033-a228-432a"
BUNNY_BASE_URL = "https://storage.bunnycdn.com/"
BUNNY_STORAGE = "lunar-research/"

TIF_SAMPLE_RATE = 0.075

DSK_FILE_CENTER_BODY_ID = 301  # NAIF ID for the Moon
DSK_FILE_SURFACE_ID = 1001  # Arbitrary surface ID

FINSCL = 6.9  # Fine voxel scale
CORSCL = 9    # Coarse voxel scale
WORKSZ = 400_000_000
VOXPSZ = 45_000_000   # Voxel-plate pointer array size
VOXLSZ = 135_000_000  # Voxel-plate list array size
SPXISZ = 1_250_000_000  # Spatial index size
MAKVTL = True

CORSYS = 3
CORPAR = np.zeros(10)
DCLASS = 2

# Undersampling rate of TIF file with elevation data
DEFAULT_TIF_SAMPLE_RATE = 0.075
# Elevation data are in tens of meters, it's divinded by this constant to obtian kilometers, might vary
TIF_TO_KM_SCALE = 100


##########################################################################################
#####                           INSTRUMENTS CONFIGURATION                            #####
##########################################################################################

# Diviner instrument IDs (from https://www.diviner.ucla.edu/instrument-specs)

IMPLICIT_BORESIGHT = np.array([0, 0, 1])

LOLA_INSTRUMENT_IDS = [-85511, -85512, -85513, -85514, -85515, -85521, -85522, -85523, -85523, -85525]

MINI_RF_CHANNELS = ["X", "S"]

LROC_NAC_IDS = [-85600, -85610]  # NAC-Left and NAC-Right
LROC_WAC_IDS = [-85631, -85632, -85633, -85634, -85635, -85641, -85642]  # VIS and UV filters

GRAIL_A_INSTRUMENTS = [-177530, -177531, -177532, -177533]
GRAIL_B_INSTRUMENTS = [-181530, -181531, -181532, -181533]

DIVINER_SUBINSTRUMENTS = [
    (-85211, 0, "INS-85205_DETECTOR_DIRS_FP_A"),
    (-85212, 1, "INS-85205_DETECTOR_DIRS_FP_A"),
    (-85213, 2, "INS-85205_DETECTOR_DIRS_FP_A"),
    (-85214, 3, "INS-85205_DETECTOR_DIRS_FP_A"),
    (-85215, 4, "INS-85205_DETECTOR_DIRS_FP_A"),
    (-85216, 5, "INS-85205_DETECTOR_DIRS_FP_A"),
    (-85221, 0, "INS-85205_DETECTOR_DIRS_FP_B"),
    (-85222, 1, "INS-85205_DETECTOR_DIRS_FP_B"),
    (-85223, 2, "INS-85205_DETECTOR_DIRS_FP_B"),
]
DIVINER_SUBINSTRUMENT_PIXEL_COUNT = 21


BORESIGHT_TO_BOUNDS_TOLERANCE_BUFFER_SIZE = 1024
# Mulitply the max discrepancy to ensure that the boresight is within the bounds
BORESIGHT_TO_BOUNDS_TOLERANCE_MULTIPLIER = 1.1
BOUNDS_TO_BORESIGHT_BUFFER_LEN = 16
