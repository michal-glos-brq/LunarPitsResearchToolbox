"""
This is the interface to our Mongo DB. Mongo is kind of exploited for our purpose just to keep
the persistant storage in one place and easily accessible on the network.

A lot of redundant and overly-specific  use cases to general use case functionalities are implemented - refactoring needed.
"""


from typing import List, Optional
import time
import threading
import pandas as pd
from pymongo import MongoClient, errors
from src.config.mongo_config import (
    MONGO_URI,
    PIT_ATLAS_PARSED_DB_NAME,
    PIT_COLLECTION_NAME,
    SIMULATION_DB_NAME,
    SIMULATION_POINTS_COLLECTION,
    RDR_DIVINER_DB,
    RDR_DIVINER_COLLECTION,
)




class Sessions:
    """
    MongoDB session manager for simulation and pit location data.

    This class provides methods to:
      - Retrieve lunar pit locations as a Pandas DataFrame.
      - Insert simulation result documents into a MongoDB timeseries collection synchronously.

    Simulation results are stored in a timeseries collection with indexes on:
      - astro_timestamp (ephemeris time, ET, as a float)
      - instrument (instrument name)
      - min_distance (the computed minimum distance)
    """
    client: MongoClient = None
    sessions = {}
    lunar_pit_locations = None
    max_retries = 100

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
            Sessions.client[db_name].create_collection("placeholder_collection")
            Sessions.client[db_name]["placeholder_collection"].drop()
        Sessions.sessions[db_name] = Sessions.client[db_name]
        return Sessions.sessions[db_name]

    @staticmethod
    def get_all_pits_points():
        """
        Fetches all lunar pit locations from the MongoDB collection and returns them as a Pandas DataFrame.

        The resulting DataFrame uses the pit 'name' as its index and includes 'latitude' and 'longitude' columns.
        """
        if Sessions.lunar_pit_locations is not None:
            return Sessions.lunar_pit_locations

        session = Sessions.get_db_session(PIT_ATLAS_PARSED_DB_NAME)
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
    def _prepare_simulation_collections(instrument_name: str, filter_str: str):
        """
        Create collection to store positive and failed simulation steps results
        """
        session = Sessions.get_db_session('spacecraft-simulation')
        positive_collection_name = f"{instrument_name}_{filter_str}"
        failed_collection_name = f"{instrument_name}_{filter_str}_failed"

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
        positive_collection.create_index("et")  # Queries by time range
        positive_collection.create_index("min_distance")  # Sorting by distance
        # collection.create_index([("boresight", "2dsphere")])  # Spatial index for boresight queries

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

        # Ensure proper indexes exist for efficient querying
        failed_collection.create_index("et")  # Queries by time range

        return positive_collection, failed_collection

    @staticmethod
    def get_simulation_collection(collection_name):
        session = Sessions.get_db_session('spacecraft-simulation')
        return session[collection_name]

    @staticmethod
    def _prepare_data_fetching_collections(dataset_name: str, treshold: float, succesfull_indices: List[str] = ["et"]):
        """
        Create collection to store positive and failed simulation steps results
        """
        session = Sessions.get_db_session('raw-data-fetching')
        positive_collection_name = f"{dataset_name}_{int(treshold)}"
        failed_collection_name = f"{dataset_name}_{int(treshold)}_failed"

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

        # Ensure proper indexes exist for efficient querying
        failed_collection.create_index("et")  # Queries by time range

        return positive_collection, failed_collection


    @staticmethod
    def background_insert_batch_timeseries_results(results: List[dict], collection):
        """
        Inserts a batch of simulation result documents into MongoDB.
        """
        if not results:
            return
        thread = threading.Thread(target=Sessions.insert_batch_timeseries_results, args=(results, collection))
        thread.daemon = True  # Kills thread if the main program exits
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
                if attempt <= Sessions.max_retries - 1:
                    raise e
                wait_time *= 1.2
                time.sleep(wait_time)

