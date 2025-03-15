"""
This file serves as a central configuration file for the MongoDB related code, saved in src/mongo
"""

import os
from src.global_config import HDD_BASE_PATH


MONGO_URI = "mongodb://admin:password@localhost:27017"


PIT_ATLAS_DB_NAME = "lro_pits"
PIT_ATLAS_PARSED_DB_NAME = "lro_pits_parsed"

PIT_ATLAS_IMAGE_COLLECTION_NAME = "images"
PIT_COLLECTION_NAME = "pits"
PIT_DETAIL_COLLECTION_NAME = "pit_details"

### Configuration for local-ran scripts (Pit Atlas scraping and parsing)
IMG_BASE_FOLDER = os.path.join(HDD_BASE_PATH, "MONGO", "PITS_IMAGES")


### Simulation data
SIMULATION_DB_NAME = "astro-simulation"
SIMULATION_POINTS_COLLECTION = "simulation_points"

RDR_DIVINER_DB = "rdr_diviner"
RDR_DIVINER_COLLECTION = "rdr_diviner_filtered" # Here, surely the querried area have to be added as a suffix
