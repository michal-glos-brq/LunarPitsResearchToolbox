"""
This is the interface to our Mongo DB. Mongo is exploited as a centralized, persistent storage accessible over the network.
It's purpose is to extract all DB work into this file

A lot of redundant and overly-specific  use cases to general use case functionalities are implemented - refactoring needed.
"""

import time
import logging
import queue
import random
import threading
from typing import List

import pandas as pd
from pymongo import MongoClient, errors

from src.db.config import (
    MONGO_URI,
    PIT_ATLAS_DB_NAME,
    PIT_COLLECTION_NAME,
    PIT_DETAIL_COLLECTION_NAME,
    PIT_ATLAS_IMAGE_COLLECTION_NAME,
    SIMULATION_DB_NAME,
    SIMULATION_METADATA_COLLECTION,
    MAX_MONGO_RETRIES,
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
    failed_inserts = queue.Queue()
    sessions = {}
    lunar_pit_locations = None

    def __init__(self):
        """
        Initialize the Sessions class.
        This is a singleton class, so the constructor should not be called directly.
        Use the static methods instead.
        """
        raise NotImplementedError("This class is a singleton and cannot be instantiated.")

    @staticmethod
    def is_client_alive(client: MongoClient):
        try:
            client.admin.command("ping")
            return True
        except errors.PyMongoError:
            return False

    @staticmethod
    def get_db_session(db_name: str):
        """
        Returns a MongoDB database session for the given database name.
        If the session does not already exist, it is created.
        If the database is new, a temporary collection is created and dropped.
        """
        for i in range(MAX_MONGO_RETRIES):
            try:
                if Sessions.client is None or not Sessions.is_client_alive(Sessions.client):
                    Sessions.client = MongoClient(
                        MONGO_URI,
                        serverSelectionTimeoutMS=5000,
                        socketTimeoutMS=10000,
                        connectTimeoutMS=5000,
                        retryWrites=True,
                        retryReads=True,
                        maxPoolSize=20,
                    )

                if hasattr(Sessions.sessions, db_name):
                    return Sessions.sessions[db_name]

                if db_name not in Sessions.client.list_database_names():
                    # Create and immediately drop a temporary collection to initialize the database.
                    Sessions.client[db_name].create_collection("_placeholder_collection")
                    Sessions.client[db_name]["_placeholder_collection"].drop()

                Sessions.sessions[db_name] = Sessions.client[db_name]
                return Sessions.sessions[db_name]
            except Exception as e:
                logging.error("Failed to connect to MongoDB: %s", e)
                if i < MAX_MONGO_RETRIES - 1:
                    time.sleep(5 * random.random())
            

    @staticmethod
    def reconnect_collection(collection):
        """
        Reconnect to the given collection object dynamically.
        """
        db_name = collection.database.name
        collection_name = collection.name
        session = Sessions.get_db_session(db_name)
        return session[collection_name]

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
                "longitude": item["location"]["coordinates"][0],
            }
            for item in query_results
        ]
        Sessions.lunar_pit_locations = pd.DataFrame(data)
        Sessions.lunar_pit_locations.set_index("name", inplace=True)
        return Sessions.lunar_pit_locations

    @staticmethod
    def prepare_simulation_collections(instrument_name: str, indices: List[str] = ["et", "meta.simulation_id"]):
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

        def create_timeseries_collection(name):
            for attempt in range(3):
                try:
                    session.create_collection(
                        name,
                        timeseries={
                            "timeField": "timestamp_utc",
                            "metaField": "meta",
                            "granularity": "seconds",
                        },
                    )
                    break  # Success
                except errors.CollectionInvalid:
                    break  # Collection already exists
                except errors.OperationFailure as e:
                    if e.code == 48:  # NamespaceExists
                        break
                    elif attempt < 2:
                        time.sleep(0.1 + random.random() * 0.2)
                        continue
                    else:
                        raise

        # Create positive and failed collections safely
        create_timeseries_collection(positive_collection_name)
        create_timeseries_collection(failed_collection_name)

        positive_collection = session[positive_collection_name]
        failed_collection = session[failed_collection_name]

        # Index creation is idempotent; safe to run always
        for index in indices:
            positive_collection.create_index(index)
            failed_collection.create_index(index)

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
        for attempt in range(MAX_MONGO_RETRIES):
            try:
                collection.insert_many(results, ordered=False)  # Insert in bulk, unordered (faster)
                return  # Success, exit function
            except errors.PyMongoError as e:
                if attempt >= MAX_MONGO_RETRIES - 1:
                    logging.error("Batch insert failed permanently: %s", e)
                    Sessions.failed_inserts.put((results, collection))
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

    @staticmethod
    def insert_simulation_metadata(metadata: dict):
        """
        Returns the simulation metadata collection.
        """
        session = Sessions.get_db_session(SIMULATION_DB_NAME)
        session[SIMULATION_METADATA_COLLECTION].insert_one(metadata)

    @staticmethod
    def update_simulation_metadata(metadata_id, current_time_iso, finished=None, metadata=None):
        update_fields = {"last_logged_time": current_time_iso}
        if finished is not None:
            update_fields["finished"] = finished
        if metadata is not None:
            update_fields["metadata"] = metadata
        session = Sessions.get_db_session(SIMULATION_DB_NAME)
        session[SIMULATION_METADATA_COLLECTION].update_one({"_id": metadata_id}, {"$set": update_fields})


    @staticmethod
    def start_background_update_simulation_metadata(metadata_id, current_time_iso, finished=None, metadata=None):
        """
        Spawns a background thread to update the simulation metadata document.
        """
        thread = threading.Thread(
            target=Sessions.update_simulation_metadata, args=(metadata_id, current_time_iso, finished, metadata)
        )
        thread.daemon = True
        thread.start()
        return thread


    @staticmethod
    def get_spacecraft_position_failed_collection(spacecraft_name: str):
        """
        Returns the failed collection for the given instrument.
        """
        session = Sessions.get_db_session(SIMULATION_DB_NAME)
        failed_collection_name = f"{spacecraft_name.replace(' ', '_')}_satellite_position_failed"
        return session[failed_collection_name]

    @staticmethod
    def process_failed_inserts():
        while not Sessions.failed_inserts.empty():
            results, collection = Sessions.failed_inserts.get()
            try:
                collection.insert_many(results, ordered=False)
                logging.info("Successfully reinserted failed batch.")
            except errors.PyMongoError as e:
                logging.error("Reinsert of failed batch failed: %s", e)
                try:
                    collection = Sessions.reconnect_collection(collection)
                    logging.info("Reconnected to collection: %s", collection.name)
                except Exception as reconnect_error:
                    logging.error("Failed to reconnect collection: %s", reconnect_error)

                Sessions.failed_inserts.put((results, collection))
                time.sleep(5)
