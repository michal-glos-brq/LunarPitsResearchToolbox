"""
This file serves as a central configuration file for the MongoDB related code, saved in src/mongo
"""

import os
from src.global_config import HDD_BASE_PATH

MASTER_ID = os.getenv("MASTER_ID", "localhost")
MONGO_URI = f"mongodb://admin:password@{MASTER_ID}:27017"

# 480-ish minutes of retrying, in case server is not reachable
MAX_MONGO_RETRIES = 240


PIT_ATLAS_DB_NAME = "lro_pits"

PIT_COLLECTION_NAME = "pits"
PIT_DETAIL_COLLECTION_NAME = "pit_details"
PIT_ATLAS_IMAGE_COLLECTION_NAME = "images"

### Configuration for local-ran scripts (Pit Atlas scraping and parsing)
IMG_BASE_FOLDER = os.path.join(HDD_BASE_PATH, "MONGO", "PITS_IMAGES")


### Simulation data
SIMULATION_DB_NAME = "astro-simulation"
SIMULATION_POINTS_COLLECTION = "simulation_points"

SIMULATION_METADATA_COLLECTION = "simulation_metadata"

RDR_DIVINER_DB = "rdr_diviner"
RDR_DIVINER_COLLECTION = "rdr_diviner_filtered" # Here, surely the querried area have to be added as a suffix
