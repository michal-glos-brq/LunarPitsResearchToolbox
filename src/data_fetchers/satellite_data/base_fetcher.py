from abc import ABC, abstractmethod
import logging
import gc
from typing import Tuple, List, Dict
from datetime import datetime, timedelta
import multiprocessing as mp
import time

from tqdm import tqdm
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.global_config import HDD_BASE_PATH, TQDM_NCOLS, MAX_DATA_DOWNLOADS, MAX_PROCESSES
from src.config import TIME_STEP
from src.db.mongo.interface import Sessions


BASE_MARGIN = 3  # Km - For lunar model misalignment


class DataFile:
    name: str
    url: str
    timewindows: List[Tuple[datetime, datetime]]


class BaseFetcher:

    def __init__(self, threshold: float, simulation_collection_name: str):
        """THreshold - how much distance from our areas of interest are we requirering data from"""
        # This threshold is used to filter localized downloaded data - Lunar model discrepacies could introduce inconsistencies
        self.threshold = threshold + BASE_MARGIN
        # This threshold is used to include the furthest bounds of the sensors (projected distance from instrument mean boresight vs furthest bound)
        self.rough_treshold = threshold + self.get_rough_treshold_margin()
        self.simulation_collection = Sessions.get_simulation_collection(simulation_collection_name)
        self.load_timewindows()
        self.load_datafiles()

    @abstractmethod
    @property
    def name(self) -> str:
        """Name of the fetcher"""
        ...

    @abstractmethod
    def load_datafiles(self):
        """
        From timeintervals of interests, url to data are composed and returned
        """
        ...

    @abstractmethod
    def process_raw_file(self, request_content: bytes, datafile: DataFile) -> List[Dict]:
        """Once the whole data file is downloaded, here is processed and filtered based on timewindows"""
        ...

    @property
    def file_download_timeout(self) -> int:
        """Can be overwritten if needed"""
        return 150

    @property
    def simulation_point_timedelta_upper_bound(self) -> timedelta:
        """Maximal time distance difference between 2 simulation steps to be considered as continuous capture"""
        return timedelta(s=(TIME_STEP * 2))  # Seems like reasonable default value

    @staticmethod
    def get_rough_treshold_margin(self):
        pipeline = [
            {"$match": {"meta.max_bound_distance": {"$exists": True}}},
            {"$group": {"_id": None, "max_max_bound_distance": {"$max": "$meta.max_bound_distance"}}},
        ]

        # Execute the aggregation
        result = list(self.collection.aggregate(pipeline))
        if len(result) == 0:
            return BASE_MARGIN
        return result[0]["max_max_bound_distance"] + BASE_MARGIN

    def load_timewindows(self):
        """Get time windows for a given collection and threshold

        The distance attribute in timeseries is rough and is computed as distance of the closest area of interest
        to projection to the lunar surface of mean instrument boresight. Ideally the treshold
        """
        timestamps = list(
            self.simulation_collection.find(
                {"distance": {"$lt": self.rough_threshold}},  # Filter condition
                {"timestamp_utc": 1, "_id": 0},  # Projection to return only timestamp_utc
            ).sort("timestamp_utc", 1)
        )  # Sorting in ascending order

        self.timestamp_list = [doc["timestamp_utc"] for doc in timestamps]

        timestamp_clusters = [[self.timestamp_list[0]]]

        for timestamp in self.timestamp_list[1:]:
            if timestamp - timestamp_clusters[-1][-1] > self.simulation_point_timedelta_upper_bound:
                timestamp_clusters.append([timestamp])
            else:
                timestamp_clusters[-1].append(timestamp)
        self.flagged_data_intervals = [(_t[0], _t[-1]) for _t in timestamp_clusters]

    def download_and_load(self, url, processing_semaphore, retries=16, backoff_factor=2):
        """Download and extract ZIP in-memory with retry mechanism."""
        # Set up a session with retry logic
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        try:
            response = session.get(url, stream=True, timeout=self.file_download_timeout)
            response.raise_for_status()

            with processing_semaphore:
                return self.process_raw_file(response.content)
        except requests.exceptions.RequestException as e:

            # TODO: Log the DB - simething went wrong to allow repeating the process

            raise Exception(f"File download failed after {retries} attempts: {e}")
        finally:
            del response

    @staticmethod
    def insert_to_db(result_queue, datafile_count: int):
        """Continuously collect processed data and insert it into MongoDB."""
        success_collection, failure_collection = Sessions._prepare_data_fetching_collections()
        threads = []

        with tqdm(total=len(datafile_count), desc="Processing dataset", ncols=TQDM_NCOLS) as pbar:
            while True:
                batch = result_queue.get()
                if batch is None:  # Exit signal
                    break

                if data := batch.get("data"):
                    threads.append(Sessions.background_insert_batch_timeseries_results(data, success_collection))
                elif error := batch.get("error"):
                    threads.append(Sessions.background_insert_batch_timeseries_results([error], failure_collection))

                pbar.update(1)  # Update progress bar

                # Kill old threads
                for thread_id in range(len(threads) - 1, -1, -1):
                    if not threads[thread_id].is_alive():
                        threads[thread_id].join()
                        threads.pop(thread_id)

        for thread in threads:
            thread.join()

    def process_datafile(self, datafile: DataFile, result_queue, download_semaphore, processing_semaphore) -> None:
        """Process and filter one data file"""
        if not datafile.timewindows:
            result_queue.put(
                {
                    "error": {
                        "timestamp_utc": datetime.now(),
                        "error": "No timewindows for datafile",
                        "meta": {"datafile": datafile.name, "url": datafile.url},
                    }
                }
            )

        try:
            with download_semaphore:
                data = self.download_and_load(datafile.url, processing_semaphore)

            if data:
                result_queue.put({"data": data})

        except Exception as e:
            result_queue.put(
                {
                    "error": {
                        "timestamp_utc": datetime.now(),
                        "error": str(e),
                        "meta": {"datafile": datafile.name, "url": datafile.url},
                    }
                }
            )
        finally:
            # We don't know when it failed, hence anything below could fail too
            try:
                del data
                gc.collect()
            except:
                ...

    def process_dataset(self):
        datafiles: List[DataFile] = self.load_datafiles()

        manager = mp.Manager()
        # Into result queue - will be put filtered data evenets and failure events
        result_queue = manager.Queue()

        # Semaphores
        download_sem = mp.BoundedSemaphore(MAX_DATA_DOWNLOADS)
        processing_sem = mp.BoundedSemaphore(MAX_PROCESSES)

        db_insert_process = mp.Process(target=BaseFetcher.insert_to_db, args=(result_queue, len(datafiles)))
        db_insert_process.start()

        processes = []
        for datafile in datafiles:
            process = mp.Process(
                target=self.process_datafile, args=(datafile, result_queue, download_sem, processing_sem)
            )
            process.start()
            processes.append(process)

            while len(processes) >= MAX_PROCESSES:
                for proc in processes:
                    if not proc.is_alive():
                        proc.join()
                        processes.remove(proc)
                time.sleep(0.1)

        # Wait for all remaining processes to finish
        for proc in processes:
            proc.join()

        # Stop DB process
        result_queue.put(None)  # Signal end of queue
        db_insert_process.join()  # Ensure DB inserts finish
