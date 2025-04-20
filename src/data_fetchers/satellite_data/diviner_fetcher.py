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

from src.db.mongo.interface import Sessions
from src.config.diviner_config import DIVINER_BASE_URL, BASE_SUFFIX

from src.data_fetchers.base_fetcher import BaseFetcher, DataFile


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
    "2024": "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2024/",
}


dtypes = {
    "orbit": "int32",
    "sundist": "float32",
    "sunlat": "float32",
    "sunlon": "float32",
    "sclk": "float64",  # Spacecraft clock - precise float
    "af": "int16",  # Activity Flag
    "c": "int8",  # Diviner Channel Number (1-9)
    "det": "int8",  # Detector Number (1-21)
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


class DivinerRDRDataFetcher(BaseFetcher):

    name: str = "DIVINER_RDR"

    def process_raw_file(self, request_content: bytes, datafile: DataFile) -> List[Dict]:
        # Extract in-memory without writing to disk
        with zipfile.ZipFile(io.BytesIO(request_content), "r") as z:
            with z.open(z.filelist[0]) as f:
                df = pd.read_csv(
                    io.StringIO(f.read().decode("utf-8")),
                    skiprows=3,
                    delimiter=",",
                    quotechar='"',
                    dtype=dtypes,
                    skipinitialspace=True,  # Removes leading spaces automatically
                    converters={"utc": lambda x: x.strip('"')},  # Removes quotes from UTC column
                )
            df.rename(columns={"#        date": "date"}, inplace=True)
            df.drop(columns=["_id"], inplace=True, errors="ignore")
            df["utc"] = pd.to_datetime(df["date"] + " " + df["utc"], format="%d-%b-%Y %H:%M:%S.%f")

            filtered_df = df[
                np.logical_or.reduce([np.logical_and(df.utc > w[0], df.utc < w[1]) for w in datafile.timewindows])
            ]
            return filtered_df.to_dict(orient="records")

    def load_datafiles(self):

        timestamp_set = set()
        for utc in tqdm(self.merge_entries_and_timewindowsimestamp_list, desc="Processing timestamps"):
            timestamp = utc.strftime("%Y%m%d%H%M")
            timestamp_set.add(f"{timestamp[:-1]}0")

        entries = sorted(timestamp_set)
        timewindows = sorted(self.flagged_data_intervals, key=lambda x: x[0])

        def data_entry_name_to_url(entry_name: str) -> str:
            base_url = YEARLY_DATA_URLS[entry_name[:4]]
            base_url = f"{base_url}/{entry_name[:6]}/{entry_name[:8]}/{entry_name}"
            return f"{base_url}_rdr.zip"

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

        return [
            DataFile(name=entry, url=data_entry_name_to_url(entry), timewindows=timewindows)
            for entry, timewindows in tqdm(sorted(list(entry_map.items()), key=lambda x: x[0]))
        ]
