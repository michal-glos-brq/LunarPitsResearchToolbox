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
from typing import List, Dict, Tuple
from datetime import datetime

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
    SIMULATION_TIME_INTERVAL_COLLECTION_NAME,
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

    ##########################################################################################
    #####                               Session management                               #####
    ##########################################################################################

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
                        # Sometimes, we read a lot ... 30 minutes seem reasonable
                        socketTimeoutMS=30 * 60 * 1000,
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

    ##########################################################################################
    #####                                Lunar Pits data                                 #####
    ##########################################################################################

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

    ##########################################################################################
    #####                             Spacecraft simulation                              #####
    ##########################################################################################

    @staticmethod
    def get_spacecraft_position_failed_collection(spacecraft_name: str):
        """
        Returns the failed collection for the given instrument.
        """
        session = Sessions.get_db_session(SIMULATION_DB_NAME)
        failed_collection_name = f"{spacecraft_name.replace(' ', '_')}_satellite_position_failed"
        return session[failed_collection_name]

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
    def prepare_simulation_intervals_collection():
        """
        Prepares the simulation intervals collection as a timeseries collection.
        Returns the prepared collection object.
        """
        session = Sessions.get_db_session(SIMULATION_DB_NAME)

        # Collection creation (idempotent)
        for attempt in range(3):
            try:
                session.create_collection(
                    SIMULATION_TIME_INTERVAL_COLLECTION_NAME,
                    timeseries={
                        "timeField": "created_at",
                        "metaField": None,
                        "granularity": "seconds",
                    },
                )
                break  # Success
            except errors.CollectionInvalid:
                break  # Already exists, OK
            except errors.OperationFailure as e:
                if e.code == 48:  # NamespaceExists
                    break
                elif attempt < 2:
                    time.sleep(0.1 + random.random() * 0.2)
                    continue
                else:
                    raise

        collection = session[SIMULATION_TIME_INTERVAL_COLLECTION_NAME]

        # Index creation (idempotent)
        collection.create_index("simulation_name")
        collection.create_index("instrument_name")
        collection.create_index("start_et")
        collection.create_index("end_et")

        return collection


    @staticmethod
    def prepare_simulation_metadata(simulation_metadata: dict) -> bool:
        """
        Checks if a finished simulation metadata record already exists that matches the following fields:
          - simulation_name
          - start_time
          - end_time
          - filter_name
          - base_step
          - instruments (matched as an unordered set, i.e. same elements and same count)
        If such a document exists, returns True (simulation already computed).
        Otherwise, inserts the provided metadata and returns False.
        """
        session = Sessions.get_db_session(SIMULATION_DB_NAME)
        collection = session[SIMULATION_METADATA_COLLECTION]
        # Query matching:
        # - "instruments": {"$all": [...], "$size": n} makes sure that the document's instruments array contains
        #   all the provided elements (regardless of order) and that its length matches exactly.
        query = {
            "simulation_name": simulation_metadata["simulation_name"],
            "start_time": simulation_metadata["start_time"],
            "end_time": simulation_metadata["end_time"],
            "filter_name": simulation_metadata["filter_name"],
            "base_step": simulation_metadata["base_step"],
            "finished": True,
            "instruments": {
                "$all": simulation_metadata["instruments"],
                "$size": len(simulation_metadata["instruments"]),
            },
        }
        existing = collection.find_one(query)
        if existing:
            return True
        else:
            collection.insert_one(simulation_metadata)
            return False

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
    def update_simulation_metadata(metadata_id, current_time_iso, finished=None, metadata=None):
        update_fields = {"last_logged_time": current_time_iso}
        if finished is not None:
            update_fields["finished"] = finished
        if metadata is not None:
            update_fields["metadata"] = metadata
        session = Sessions.get_db_session(SIMULATION_DB_NAME)
        session[SIMULATION_METADATA_COLLECTION].update_one({"_id": metadata_id}, {"$set": update_fields})

    @staticmethod
    def simulation_tasks_query(filter_name: str, simulation_name: str, instrument_names: List[str]) -> List[dict]:
        """
        Queries the simulation metadata collection for tasks that match the specified filter_name,
        simulation_name, and list of instrument_names. The results are sorted by the "start_time" field.

        Parameters:
          - filter_name (str): The name of the filter used in the simulation.
          - simulation_name (str): The experiment/simulation name.
          - instrument_names (list[str]): List of instrument names (must be an exact unordered match).

        Returns:
          - List[dict]: A list of simulation metadata documents sorted by "start_time" in ascending order.
        """
        # Obtain a session for the simulation database.
        session = Sessions.get_db_session(SIMULATION_DB_NAME)
        # Get the simulation metadata collection.
        collection = session[SIMULATION_METADATA_COLLECTION]

        # Build the query.
        query = {
            "simulation_name": simulation_name,
            "filter_name": filter_name,
            "finished": True,  # Optionally query only finished simulations.
            "instruments": {
                "$all": instrument_names,
                "$size": len(instrument_names),
            },
        }

        # Execute the query and sort the results by the 'start_time' field (ascending order).
        cursor = collection.find(query).sort("start_time", 1)

        # Convert the cursor to a list of documents.
        simulation_task_documents = list(cursor)
        return simulation_task_documents

    @staticmethod
    def insert_simulation_intervals(simulation_name: str, intervals_dict: Dict[str, List[Tuple[float, float]]]):
        """
        Inserts simulation time intervals into MongoDB timeseries collection.
        Each interval is stored as a separate document.
        """
        collection = Sessions.prepare_simulation_intervals_collection()
        now = datetime.utcnow()

        documents = []
        for instrument_name, intervals in intervals_dict.items():
            # Delete existing entries for this simulation + instrument
            delete_result = collection.delete_many({
                "simulation_name": simulation_name,
                "instrument_name": instrument_name,
            })
            logging.info(
                f"Deleted {delete_result.deleted_count} existing interval documents "
                f"for simulation '{simulation_name}' and instrument '{instrument_name}'."
            )

            for start_et, end_et in intervals:
                documents.append({
                    "simulation_name": simulation_name,
                    "instrument_name": instrument_name,
                    "created_at": now,
                    "start_et": start_et,
                    "end_et": end_et,
                })

        if documents:
            collection.insert_many(documents, ordered=False)
            logging.info(f"Inserted {len(documents)} interval documents into '{collection.name}'.")
        else:
            logging.warning("No intervals to insert.")

    ##########################################################################################
    #####                               Background Tasks                                 #####
    ##########################################################################################

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
