"""
============================================================
Dataclass for instrument projection onto a body surface
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

This file serves as a central configuration file for the MongoDB related code, saved in src/mongo
"""

import os
from src.global_config import HDD_BASE_PATH

MASTER_IP = os.getenv("MASTER_IP", "host.docker.internal")
MONGO_URI = f"mongodb://admin:password@{MASTER_IP}:27017"

# A lot of minutes, will wait to reconnect just in case
MAX_MONGO_RETRIES = 4096


PIT_ATLAS_DB_NAME = "lro_pits"

PIT_COLLECTION_NAME = "pits"
PIT_DETAIL_COLLECTION_NAME = "pit_details"
PIT_ATLAS_IMAGE_COLLECTION_NAME = "images"

SIMULATION_TIME_INTERVAL_COLLECTION_NAME = "simulation_time_intervals"

### Configuration for local-ran scripts (Pit Atlas scraping and parsing)
IMG_BASE_FOLDER = os.path.join(HDD_BASE_PATH, "MONGO", "PITS_IMAGES")


### Simulation data
#SIMULATION_DB_NAME = "simulationDB"
SIMULATION_DB_NAME = "simulations"

SIMULATION_POINTS_COLLECTION = "simulation_points"

SIMULATION_METADATA_COLLECTION = "simulation_metadata"

EXTRACTOR_DB_NAME = "extractorDB"
EXTRACTOR_METADATA_COLLECTION = "extractor_metadata"


RDR_DIVINER_DB = "rdr_diviner"
RDR_DIVINER_COLLECTION = "rdr_diviner_filtered"
