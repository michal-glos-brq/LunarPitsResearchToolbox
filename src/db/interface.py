"""
This is the interface to our Mongo DB. Mongo is exploited as a centralized, persistent storage accessible over the network.
It's purpose is to extract all DB work into this file

A lot of redundant and overly-specific  use cases to general use case functionalities are implemented - refactoring needed.
"""


from typing import List
import time
import threading
import pandas as pd
from pymongo import MongoClient, errors
from src.db.config import (
    MONGO_URI,
    PIT_ATLAS_DB_NAME,
    PIT_COLLECTION_NAME,
    PIT_DETAIL_COLLECTION_NAME,
    PIT_ATLAS_IMAGE_COLLECTION_NAME,
    SIMULATION_DB_NAME,
)


class Sessions:
    """
    MongoDB session manager for simulation results and lunar pit atlas data.
    
    This class provides a centralized interface to:
      - Access MongoDB databases for parsed and raw lunar pit data (pits, pit details, images).
      - Fetch all parsed lunar pit locations as a Pandas DataFrame (with caching).
      - Initialize and manage timeseries collections for simulation results (positive and failed).
      - Perform threaded, batched inserts of simulation results into MongoDB with retry logic.
    
    Notes:
      - Simulation collections use MongoDB's timeseries format, with `timestamp_utc` as the time field
        and `meta` as the metadata field.
      - All database and collection names are defined in `src.db.config`.
      - This class is intentionally non-modular and purpose-built for fast iteration in scientific pipelines.
    """

    client: MongoClient = None
    sessions = {}
    lunar_pit_locations = None
    # It will eventually go thoutgh ...
    max_retries = 100


    def __init__(self):
        """
        Initialize the Sessions class.
        This is a singleton class, so the constructor should not be called directly.
        Use the static methods instead.
        """
        raise NotImplementedError("This class is a singleton and cannot be instantiated.")

    @staticmethod
    def get_db_session(db_name: str):
        """
        Returns a MongoDB database session for the given database name.
        If the session does not already exist, it is created.
        If the database is new, a temporary collection is created and dropped.
        """
        if Sessions.client is None:
            Sessions.client = MongoClient(MONGO_URI)

        if hasattr(Sessions.sessions, db_name):
            return Sessions.sessions[db_name]

        if db_name not in Sessions.client.list_database_names():
            # Create and immediately drop a temporary collection to initialize the database.
            Sessions.client[db_name].create_collection("_placeholder_collection")
            Sessions.client[db_name]["_placeholder_collection"].drop()
        Sessions.sessions[db_name] = Sessions.client[db_name]

        return Sessions.sessions[db_name]

    @staticmethod
    def get_all_pits_points():
        """
        Fetches all lunar pit locations from the parsed MongoDB collection
        and returns them as a Pandas DataFrame, with pit 'name' as index
        and columns for 'latitude' and 'longitude'.
        """
        if Sessions.lunar_pit_locations is not None:
            return Sessions.lunar_pit_locations

        session = Sessions.get_db_session(PIT_ATLAS_DB_NAME)
        collection = session[PIT_COLLECTION_NAME]
        query_results = list(collection.find({}, {"location": 1, "name": 1}))
        data = [
            {
                "name": item["name"],
                "latitude": item["location"]["coordinates"][1],
                "longitude": item["location"]["coordinates"][0]
            }
            for item in query_results
        ]
        Sessions.lunar_pit_locations = pd.DataFrame(data)
        Sessions.lunar_pit_locations.set_index("name", inplace=True)
        return Sessions.lunar_pit_locations

    @staticmethod
    def prepare_simulation_collections(instrument_name: str, succesfull_indices: List[str] = ["et"]):
        """
        Ensures collections exist to store positive and failed simulation results for the given instrument.
        Returns a tuple: (positive_collection, failed_collection).

        params:
        instrument_name (str): The name of the instrument for which collections are created.
        succesfull_indices (list): List of indices to create for the positive collection.
        """
        session = Sessions.get_db_session(SIMULATION_DB_NAME)
        positive_collection_name = f"{instrument_name}"
        failed_collection_name = f"{instrument_name}_failed"

        if positive_collection_name not in session.list_collection_names():
            try:
                session.create_collection(
                    positive_collection_name,
                    timeseries={
                        "timeField": "timestamp_utc",  # Ephemeris time as float
                        "metaField": "meta",  # Optional metadata
                        "granularity": "seconds"  # Adjust based on data density
                    }
                )
            except errors.CollectionInvalid:
                pass  # Collection already exists

        positive_collection = session[positive_collection_name]

        # Ensure proper indexes exist for efficient querying
        for index in succesfull_indices:
            positive_collection.create_index(index)
        # collection.create_index([("boresight", "2dsphere")])  # Spatial index for boresight queries
        # Optionally: positive_collection.create_index([("boresight", "2dsphere")]) for spatial queries.
        if failed_collection_name not in session.list_collection_names():
            try:
                session.create_collection(
                    failed_collection_name,
                    timeseries={
                        "timeField": "timestamp_utc",  # Ephemeris time as float
                        "metaField": "meta",  # Optional metadata
                        "granularity": "seconds"  # Adjust based on data density
                    }
                )
            except errors.CollectionInvalid:
                pass

        failed_collection = session[failed_collection_name]
        failed_collection.create_index("et")  # Queries by time range

        return positive_collection, failed_collection


    @staticmethod
    def start_background_batch_insert(results: List[dict], collection):
        """
        Spawns a background thread to insert a batch of simulation results into MongoDB.
        """
        if not results:
            return
        thread = threading.Thread(target=Sessions.insert_batch_timeseries_results, args=(results, collection))
        thread.daemon = True  # The thread will exit if the main program exits.
        thread.start()
        return thread

    @staticmethod
    def insert_batch_timeseries_results(results: List[dict], collection):
        """
        Inserts a batch of simulation result documents into MongoDB.
        """
        if not results:
            return  # No data to insert

        wait_time = 2

        # Retry loop with exponential backoff
        for attempt in range(Sessions.max_retries):
            try:
                collection.insert_many(results, ordered=False)  # Insert in bulk, unordered (faster)
                return  # Success, exit function
            except errors.PyMongoError as e:
                if attempt >= Sessions.max_retries - 1:
                    raise e
                wait_time *= 1.2
                time.sleep(wait_time)


    @staticmethod
    def get_lunar_pit_collections(db_name=PIT_ATLAS_DB_NAME):
        """
        Returns a list of all lunar pit collections in the database.
        If parsed is True, it returns collections from the parsed database.
        Otherwise, it returns collections from the original database.

        Returns a tuple of (pits, pit-details, image) collections
        """
        session = Sessions.get_db_session(db_name)
        return (
            session[PIT_COLLECTION_NAME],
            session[PIT_DETAIL_COLLECTION_NAME],
            session[PIT_ATLAS_IMAGE_COLLECTION_NAME],
        )
