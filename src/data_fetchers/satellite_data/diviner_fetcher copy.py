import sys

sys.path.insert(0, "/home/mglos/skola/DIP/LavaTubeSniffer")

import time
from datetime import datetime
import os
import logging
import gc
import multiprocessing as mp
import zipfile
from datetime import timedelta
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io


### TechDebt ++ Everything is soooo clumsy!

from src.db.mongo.interface import Sessions
from src.config.diviner_config import DIVINER_BASE_URL, BASE_SUFFIX

SOFT_TRESHOLD = 12.5


MAX_PROCESSING = 5
MAX_DOWNLOADS = 20
MAX_PROCESSES = 28

BASE_HDD_PATH = "/media/mglos/HDD_8TB3/SPICE/DIVINER_DATA/"

DATASET_ROOT_URL = f"{DIVINER_BASE_URL}{BASE_SUFFIX}"
YEARLY_DATA_URLS = {
    "2009": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2009/",
    "2010": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2010/",
    "2011": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2011/",
    "2012": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2012/",
    "2013": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2013/",
    "2014": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2014/",
    "2015": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2015/",
    "2016": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2016/",
    "2017": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2017/",
    "2018": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2018/",
    "2019": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2019/",
    "2020": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2020/",
    "2021": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2021/",
    "2022": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2022/",
    "2023": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2023/",
    "2024": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2024/"
}


dtypes = {
    "orbit": "int32",
    "sundist": "float32",
    "sunlat": "float32",
    "sunlon": "float32",
    "sclk": "float64",  # Spacecraft clock - precise float
    "sclat": "float32",
    "sclon": "float32",
    "scrad": "float32",  # Distance from Moon center
    "scalt": "float32",  # Spacecraft altitude
    "el_cmd": "float16",  # Elevation command
    "az_cmd": "float16",  # Azimuth command
    "af": "int16",  # Activity Flag
    "orientlat": "float32",
    "orientlon": "float32",
    "c": "int8",  # Diviner Channel Number (1-9)
    "det": "int8",  # Detector Number (1-21)
    "vlookx": "float32",
    "vlooky": "float32",
    "vlookz": "float32",
    "radiance": "float32",  # Radiance (W/m²/sr)
    "tb": "float32",  # Brightness Temperature (K)
    "clat": "float32",  # Latitude of FOV center
    "clon": "float32",  # Longitude of FOV center
    "cemis": "float32",  # Emission Angle
    "csunzen": "float32",  # Solar zenith angle
    "csunazi": "float32",  # Solar azimuth angle
    "cloctime": "float32",  # Local time
    "qca": "int8",  # Quality control flag
    "qge": "int8",
    "qmi": "int8",
}



def download_and_extract(url, retries=10, backoff_factor=2):
    """Download and extract ZIP in-memory with retry mechanism."""
        # Set up a session with retry logic
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.get(url, stream=True, timeout=150)
        response.raise_for_status()

        # Extract in-memory without writing to disk
        with zipfile.ZipFile(io.BytesIO(response.content), 'r') as z:
            with z.open(z.filelist[0]) as f:
                return io.StringIO(f.read().decode("utf-8"))
    except requests.exceptions.RequestException as e:
        raise Exception(f"File download failed after {retries} attempts: {e}")
    finally:
        del response

def process_single_entry(entry: str, timewindows: List[Tuple[datetime, datetime]], result_queue: mp.Queue, download_semaphore, processing_semaphore):
    if not timewindows:
        logging.warning(f"No timewindows for {entry}.")
        result_queue.put({})
        return

    try:
        urls = DivinerRDRDownloader.data_fragment_name_to_url(entry)
        with download_semaphore:
            tab_file = download_and_extract(urls['zip'])

        with processing_semaphore:
            df = pd.read_csv(
                tab_file,
                skiprows=3,
                delimiter=",",
                quotechar='"',
                dtype=dtypes,
                skipinitialspace=True,  # Removes leading spaces automatically
                converters={"utc": lambda x: x.strip('"')}  # Removes quotes from UTC column
            )
            df.rename(columns={"#        date": "date"}, inplace=True)
            df.drop(columns=["_id"], inplace=True, errors="ignore")
            df["utc"] = pd.to_datetime(df["date"] + " " + df["utc"], format="%d-%b-%Y %H:%M:%S.%f")


            filtered_df = df[np.logical_or.reduce([np.logical_and(df.utc > w[0] , df.utc < w[1]) for w in timewindows])]
            result_queue.put(filtered_df.to_dict(orient='records'))
            # os.remove(zip_download_path)
                # result_queue.put(df[np.logical_or.reduce([np.logical_and(df.utc > window[0], df.utc < window[1]) for window in timewindows])].to_dict(orient='records'))
    except Exception as e:
        logging.error(f"Processing failed for {entry}: {e}")
        result_queue.put([])
    finally:
        try:
            del df
            del tab_file
            del filtered_df
            gc.collect()
        except: ...


class DivinerRDRDownloader:


    def __init__(self):
        session = Sessions.get_db_session('astro-simulation')
        collection = session['simulation_points_DIVINER_test_full']

        threshold = 12.5  # Set your threshold value

        # Query only timestamp_utc where min_distance is less than the threshold
        timestamps = list(collection.find(
            {"min_distance": {"$lt": threshold}},  # Filter condition
            {"timestamp_utc": 1, "_id": 0}  # Projection to return only timestamp_utc
        ).sort("timestamp_utc", 1))  # Sorting in ascending order

        timestamp_list = [doc["timestamp_utc"] for doc in timestamps]
        timestamp_set = set()

        for utc in tqdm(timestamp_list, desc="Processing timestamps"):
            timestamp = utc.strftime("%Y%m%d%H%M")
            timestamp_set.add(f"{timestamp[:-1]}0")

        self.flagged_data_fragments = sorted(timestamp_set)
        os.makedirs(BASE_HDD_PATH, exist_ok=True)

        diff = timedelta(seconds=1.4)
        timestamp_clusters = [[timestamp_list[0]]]

        for timestamp in tqdm(timestamp_list[1:], desc="Extracting timestamp intervals"):
            if timestamp - timestamp_clusters[-1][-1] > diff:
                timestamp_clusters.append([timestamp])
            else:
                timestamp_clusters[-1].append(timestamp)
        self.flagged_data_intervals = [(_t[0], _t[-1]) for _t in timestamp_clusters]
        self._timestamp_clusters = timestamp_clusters

        session.client.close()
        del session, collection



    @staticmethod
    def data_fragment_name_to_url(fragment_name: str) -> str:
        base_url = YEARLY_DATA_URLS[fragment_name[:4]]
        base_url = f"{base_url}/{fragment_name[:6]}/{fragment_name[:8]}/{fragment_name}"
        return {
            'xml': f"{base_url}_rdr.xml",
            'zip': f"{base_url}_rdr.zip",
            'lbl': f"{base_url}_rdr.lbl",
        }


    @staticmethod
    def merge_entries_and_timewindows(entries: List[str], timewindows: List[Tuple[datetime, datetime]]):
        delta = timedelta(minutes=10)
        entry_map = {entry: [] for entry in entries}

        for window in tqdm(timewindows, desc="Assigning intervals to dataset fragmenrs"):
            min_time_string = window[0].strftime("%Y%m%d%H%M")[:-1] + "0"
            max_time_string = window[1].strftime("%Y%m%d%H%M")[:-1] + "0"
            
            entry_map[min_time_string].append(window)

            while min_time_string != max_time_string:
                min_time_string = (window[0] + delta).strftime("%Y%m%d%H%M")[:-1] + "0"
                entry_map[min_time_string].append(window)

        for entry in tqdm(entries, desc="Sorting timewindows"):
            entry_map[entry].sort(key=lambda x: x[0])

        return [(fragment, windows)for fragment, windows in tqdm(sorted(list(entry_map.items()), key=lambda x: x[0]))]



    @staticmethod
    def insert_to_db(result_queue, tasks_params):
        """Continuously collect processed data and insert it into MongoDB."""
        collection = Sessions._prepare_DIVINER_RDR_filtered_data_collection()
        with tqdm(total=len(tasks_params), desc="Processing dataset", ncols=160) as pbar:
            while True:
                batch = result_queue.get()
                if batch is None:  # Exit signal
                    break

                if batch:
                    Sessions.insert_DIVINER_RDR_dataset_entries(batch, collection)
                else:
                    logging.warning("Failed to process a batch.")
                pbar.update(1)  # Update progress bar


    def process_dataset(self):
        entries = sorted(self.flagged_data_fragments)
        timewindows = sorted(self.flagged_data_intervals, key=lambda x: x[0])

        tasks_params = DivinerRDRDownloader.merge_entries_and_timewindows(entries, timewindows)
        manager = mp.Manager()
        result_queue = manager.Queue()
    
        # Semaphores
        download_sem = mp.BoundedSemaphore(MAX_DOWNLOADS)
        processing_sem = mp.BoundedSemaphore(MAX_PROCESSES)

        db_insert_process = mp.Process(target=DivinerRDRDownloader.insert_to_db, args=(result_queue, tasks_params))
        db_insert_process.start()

        processes = []
        for entry, windows in tasks_params:
            process = mp.Process(target=process_single_entry, args=(entry, windows, result_queue, download_sem, processing_sem))
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


if __name__ == "__main__":
    downloader = DivinerRDRDownloader()
    # downloader.flagged_data_fragments = sorted(downloader.flagged_data_fragments, key=lambda x: x[0])[:64]
    downloader.process_dataset()