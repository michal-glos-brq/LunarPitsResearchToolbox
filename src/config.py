import numpy as np

###############################################
#             SPICE METADATA                  #
###############################################

# Use more precise frame
#LUNAR_FRAME = "MOON_ME"
LUNAR_FRAME = "MOON_PA_DE440"

LUNAR_TIF_DATA_URL = "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

MAX_LOADED_DYNAMIC_KERNELS = 3

SPICE_FOLDER = "/media/mglos/HDD_8TB2/SPICE"

LRO_SPICE_KERNEL_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/"
GRAIL_SPICE_KERNEL_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/"
#SELENE_SPICE_KERNEL_BASE_URL = "https://darts.isas.jaxa.jp/pub/spice/SELENE/kernels_ORG/"
#CHANDRAYAAN_SPICE_KERNEL_BASE_URL = ""
GENERIC_SPICE_KERNEL_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"


###############################################
#           DSK CREATION CONSTANTS            #
###############################################

DSK_FILE_CENTER_BODY_ID = 301  # NAIF ID for the Moon
DSK_FILE_SURFACE_ID = 1001  # Arbitrary surface ID

# Configure carefully ...
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




###############################################
#         TRAJECTORY SIMULATION DATA          #
###############################################

# It's a little bit less, used for radius correction
LRO_SPEED = 1.7
QUERY_RADIUS_MULTIPLIER = 1.3


###############################################
#              INSTRUMENT STUFF               #
###############################################

# Diviner instrument IDs (from https://www.diviner.ucla.edu/instrument-specs)

LOLA_INSTRUMENT_IDS = [-85511, -85512, -85513, -85514, -85515, -85521, -85522, -85523, -85523, -85525]
MINIRF_INSTRUMENT_IDS = [-85700]
WAC_INSTRUMENT_IDS = [-85621, -85626]
NAC_INSTRUMENT_IDS = [-85600, -85610]


LRO_DIVINER_FRAME_STR_ID = "LRO_DLRE"
LRO_LOLA_FRAME_STR_ID = "LRO_LOLA"
LRO_MINIRF_FRAME_STR_ID = "LRO_MINIRF"



DIVINER_INSTRUMENT_ID_TO_RDR_INDEX = {
    -85211: 1,
    -85212: 2,
    -85213: 3,
    -85214: 4,
    -85215: 5,
    -85216: 6,
    -85221: 7,
    -85222: 8,
    -85223: 9,
}




MOON_STR_ID = "MOON"
MOON_REF_FRAME_STR_ID = "MOON_ME"
LRO_STR_ID = "LUNAR RECONNAISSANCE ORBITER"

# Additional simulation configuration
ABBERRATION_CORRECTION = "CN+S"
TIME_STEP = 1.024  # Step through trajectory computation in seconds
MAX_TIME_STEP = 3600
MAX_LOADED_SPICE = 3  # Maximum number of dynamic SPICE kernels loaded at once
