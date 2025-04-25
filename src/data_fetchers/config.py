import os

from src.global_config import HDD_BASE_PATH

MONGO_UPLOAD_BATCH_SIZE = 1024


LOLA_LBL_FILE_DUMP = os.path.join(HDD_BASE_PATH, "LOLA_LBL")
LOLA_REMOTE_CACHE_FILE = "cache.pkl"

LOLA_BASE_URL = "https://pds-geosciences.wustl.edu"
LOLA_DATASET_ROOT = "/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_rdr/"
