import os

from astropy.time import Time

from src.global_config import HDD_BASE_PATH

MONGO_UPLOAD_BATCH_SIZE = 1024

# How long between dumping the extraction state to stdout
EXTR_STATE_DUMP_INTERVAL = 120

##### Mini-RF #####
MINIRF_BASE_URL = "https://pds-geosciences.wustl.edu/"

MINNIRF_REMOTE_CACHE_FILE = "minirf_data_structure_parsed.pkl"

MINI_RF_URLS = [
    "/lro/lro-l-mrflro-4-cdr-v1/lromrf_0001/data/sar/",
    "/lro/lro-l-mrflro-4-cdr-v1/lromrf_0002/data/sar/",
    "/lro/lro-l-mrflro-4-cdr-v1/lromrf_0003/data/sar/",
    "/lro/lro-l-mrflro-4-cdr-v1/lromrf_0004/data/sar/",
    "/lro/lro-l-mrflro-4-cdr-v1/lromrf_0005/data/sar/",
]

MINI_RF_MODES = {
    "BASELINE_S": 0,
    "ZOOM_S": 1,
    "BASELINE_X": 2,
    "ZOOM_X": 3,
}

MINI_RF_S_MODES_KEYS = ["BASELINE_S", "ZOOM_S"]
MINI_RF_X_MODES_KEYS = ["BASELINE_X", "ZOOM_X"]
MINI_RF_S_MODES_IDX = [MINI_RF_MODES[mode] for mode in MINI_RF_S_MODES_KEYS]
MINI_RF_X_MODES_IDX = [MINI_RF_MODES[mode] for mode in MINI_RF_X_MODES_KEYS]


##### DIVINER #####
MAX_DATA_METADATA_PARALLEL_DOWNLOADS = 16
DIVINER_DATA_START = Time("2009-07-05T16:50:26.195", format="isot", scale="utc")
DIVINER_DATA_END = Time("2024-12-16T00:00:00.027", format="isot", scale="utc")
DIVINER_BASE_URL = "https://pds-geosciences.wustl.edu"

DIVINER_YEARLY_DATA_URLS = {
    2009: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2009/",
    2010: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2010/",
    2011: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2011/",
    2012: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2012/",
    2013: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2013/",
    2014: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2014/",
    2015: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2015/",
    2016: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2016/",
    2017: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2017/",
    2018: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2018/",
    2019: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2019/",
    2020: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2020/",
    2021: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2021/",
    2022: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2022/",
    2023: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2023/",
    2024: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2024/",
}


##### LOLA #####
LOLA_LBL_FILE_DUMP = os.path.join(HDD_BASE_PATH, "LOLA_LBL")
LOLA_REMOTE_CACHE_FILE = "cache.pkl"

LOLA_BASE_URL = "https://pds-geosciences.wustl.edu"
LOLA_DATASET_ROOT = "/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/"

LOLA_COL_SPECS = (
    [
        ("MET_SECONDS", 4, True),
        ("SUBSECONDS", 4, False),
        # TRANSMIT_TIME: 8 bytes total = 2 × 4-byte words
        ("TRANSMIT_TIME", 4, False, 2),
        ("LASER_ENERGY", 4, True),
        ("TRANSMIT_WIDTH", 4, True),
        ("SC_LONGITUDE", 4, True),
        ("SC_LATITUDE", 4, True),
        ("SC_RADIUS", 4, False),
        ("SELENOID_RADIUS", 4, False),
    ]
    + [
        # Spots 1–5
        (f"{fld}_{i}", width, signed)
        for i in range(1, 6)
        for fld, width, signed in [
            ("LONGITUDE", 4, True),
            ("LATITUDE", 4, True),
            ("RADIUS", 4, True),
            ("RANGE", 4, False),
            ("PULSE", 4, True),
            ("ENERGY", 4, False),
            ("BACKGROUND", 4, False),
            ("THRESHOLD", 4, False),
            ("GAIN", 4, False),
            ("SHOT_FLAG", 4, False),
        ]
    ]
    + [
        # trailing half-words and words
        (
            "OFFNADIR_ANGLE",
            2,
            False,
        ),  # COLUMN 60 :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
        (
            "EMISSION_ANGLE",
            2,
            False,
        ),  # COLUMN 61 :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
        (
            "SOLAR_INCIDENCE",
            2,
            False,
        ),  # COLUMN 62 :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}
        (
            "SOLAR_PHASE",
            2,
            False,
        ),  # COLUMN 63 :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9}
        (
            "EARTH_RANGE",
            4,
            False,
        ),  # COLUMN 64 :contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11}
        (
            "EARTH_PULSE",
            2,
            False,
        ),  # COLUMN 65 :contentReference[oaicite:12]{index=12}&#8203;:contentReference[oaicite:13]{index=13}
        (
            "EARTH_ENERGY",
            2,
            False,
        ),  # COLUMN 66 :contentReference[oaicite:14]{index=14}&#8203;:contentReference[oaicite:15]{index=15}
    ]
)
