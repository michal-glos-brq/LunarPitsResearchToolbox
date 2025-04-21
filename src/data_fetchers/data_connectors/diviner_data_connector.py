"""
WARNING: When assigning tasks, we are looking for separate day datasets, but there are some 0.1s overlaps. If you
do not want to risk data loss by omission, assign the tasks with times in the middle of the day
"""

import zipfile
import pvl
from pvl.decoder import PVLDecoder
import io
import logging
from pprint import pformat
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import Future
from urllib.parse import urlparse, urljoin

import pandas as pd
import numpy as np
from tqdm import tqdm
from cachetools import LRUCache
from bs4 import BeautifulSoup as bs
import spiceypy as spice
from astropy.time import Time

from src.data_fetchers.data_connectors.virtual_file import VirtualFile
from src.data_fetchers.interval_manager import TimeInterval, IntervalList
from src.data_fetchers.data_connectors.base_data_connector import BaseDataConnector
from src.SPICE.config import LRO_INT_ID
from src.global_config import SUPRESS_TQDM
from src.SPICE.filters import BaseFilter
from src.SPICE.instruments.instrument import BaseInstrument


class NoDatetimeDecoder(PVLDecoder):
    def decode_datetime(self, value):
        return spice.utc2et(value)


# logger = logging.getLogger(__name__)
logger = logging.getLogger("kokos")
logger.setLevel(logging.INFO)

### Eventually check the website for updates, dynamic parsing is overkill in this instance

DATA_START = Time("2009-07-05T16:50:26.195", format="isot", scale="utc")
DATA_END = Time("2024-12-16T00:00:00.027", format="isot", scale="utc")

YEARLY_DATA_URLS = {
    2009: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2009/",
    2010: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2010/",
    2011: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2011/",
    2012: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2012/",
    2013: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2013/",
    2014: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2014/",
    2015: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2015/",
    2016: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1001/data/2016/",
    2017: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2017/",
    2018: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2018/",
    2019: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2019/",
    2020: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2020/",
    2021: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2021/",
    2022: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2022/",
    2023: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2023/",
    2024: "https://pds-geosciences.wustl.edu/lro/lro-l-dlre-4-rdr-v1/lrodlr_1002/data/2024/",
}

BASE_URL = "https://pds-geosciences.wustl.edu"

MAX_DATA_METADATA_PARALLEL_DOWNLOADS = 16

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


spacecraft_position_cache = LRUCache(maxsize=128)


class DivinerDataConnector(BaseDataConnector):

    name = "DivinerRDR"

    timeseries = {
        "timeField": "et",
        "metaField": "meta",
        "granularity": "seconds",
    }

    indices = [
        "et",  # always index your time field
        "orbit",  # if you ever query by orbit
        "c",  # channel
        "det",  # detector
        "cloctime",  # Local time (helps with diurnal effects)
        "csunazi",  # Solar azimuth
        "csunzen",  # Solar zenith
        "cemis",  # Emission angle
        "cx_projected", # Projected coordinates
        "cy_projected",
        "cz_projected",
        "meta._simulation_name",  # if you ever query by simulation name
    ]

    def discover_year_urls(self, min_time: Time, max_time: Time) -> Dict[Tuple[int, int], str]:
        """
        Discover the URLs of the years that are in the given time intervals.
        :param time_intervals: Time intervals to check
        :return: List of URLs for each year
        """
        max_year = max_time.datetime.year
        min_year = min_time.datetime.year

        logger.info(f"Discovering year URLs for {min_year} to {max_year}")

        tasks = []
        for current_year in range(min_year, max_year + 1):
            url = urlparse(YEARLY_DATA_URLS[current_year])
            tasks.append((current_year, url, self.fetch_url_async(url.geturl())))

        month_urls: Dict[Tuple[int, int], str] = {}

        for year, url, task in tasks:
            response = task.result()
            soup = bs(response.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and href.startswith(url.path):
                    month_urls[(year, int(href[-3:-1]))] = urljoin(BASE_URL, href)

        # Drop the months that are not in the time intervals
        for year, month in list(month_urls.keys()):
            if year == min_year and month < min_time.datetime.month:
                del month_urls[(year, month)]
            elif year == max_year and month > max_time.datetime.month:
                del month_urls[(year, month)]

        logger.info(f"Found {len(month_urls)} month URLs")
        logger.debug(f"Month URLs: \n{pformat(month_urls)}")

        return month_urls

    def discover_month_urls(
        self, min_time: Time, max_time: Time, monthly_urls: Dict[Tuple[int, int], str]
    ) -> Dict[Tuple[int, int, int], str]:
        """
        Discover the URLs of the months that are in the given time intervals.
        :param time_intervals: Time intervals to check
        :return: List of URLs for each month
        """
        tasks = []

        logger.info(
            f"Discovering month URLs for {min_time.datetime.month}-{min_time.datetime.year} to {max_time.datetime.month}-{max_time.datetime.year}"
        )

        for (year, month), url in monthly_urls.items():
            url = urlparse(url)
            tasks.append((year, month, url, self.fetch_url_async(url.geturl())))

        daily_urls: Dict[Tuple[int, int, int], str] = {}

        for year, month, url, task in tasks:
            response = task.result()
            soup = bs(response.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and href.startswith(url.path):
                    daily_urls[(year, month, int(href[-3:-1]))] = urljoin(BASE_URL, href)

        # Drop the days that are not in the time intervals
        for year, month, day in list(daily_urls.keys()):
            if year == min_time.datetime.year and month == min_time.datetime.month and day < min_time.datetime.day:
                del daily_urls[(year, month, day)]
            elif year == max_time.datetime.year and month == max_time.datetime.month and day > max_time.datetime.day:
                del daily_urls[(year, month, day)]

        logger.info(f"Found {len(daily_urls)} daily URLs")
        logger.debug(f"Daily URLs: \n{pformat(daily_urls)}")

        return daily_urls

    def discover_day_urls(self, daily_urls: Dict[Tuple[int, int, int], str]) -> Dict[Tuple[int, int, int, int], str]:
        """
        Discover the URLs of the days that are in the given time intervals.
        :param time_intervals: Time intervals to check
        :return: List of URLs for each day
        """
        tasks = []
        for (year, month, day), url in daily_urls.items():
            url = urlparse(url)
            tasks.append((year, month, day, url, self.fetch_url_async(url.geturl())))

        minute_urls: Dict[Tuple[int, int, int, int, int], str] = defaultdict(dict)

        for year, month, day, url, task in tasks:
            response = task.result()
            soup = bs(response.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and href.startswith(url.path):
                    # Parse hours and minutes too, use file suffix as key to the other dict
                    minute_urls[(year, month, day, int(href[-12:-10]), int(href[-10:-8]))][href[-3:].lower()] = urljoin(
                        BASE_URL, href
                    )

        logger.info(f"Found {len(minute_urls)} minute URLs")
        logger.debug(f"Minute URLs: \n{pformat(minute_urls)}")

        return minute_urls

    def discover_files(self, time_intervals: IntervalList):
        min_time = time_intervals.start_astropy_time
        max_time = time_intervals.end_astropy_time

        month_urls = self.discover_year_urls(min_time, max_time)
        daily_urls = self.discover_month_urls(min_time, max_time, month_urls)
        minute_urls = self.discover_day_urls(daily_urls)

        lbl_tasks: List[Tuple[Dict[str, str], Future]] = []
        for file_dict in minute_urls.values():
            lbl_tasks.append((file_dict, self.fetch_url_async(file_dict["lbl"])))

        virtual_files: List[VirtualFile] = []
        for file_dict, fut in tqdm(lbl_tasks, desc="Downloading metadata", disable=SUPRESS_TQDM):
            resp = fut.result()
            meta = pvl.loads(resp.text, decoder=NoDatetimeDecoder())
            interval = TimeInterval(meta["UNCOMPRESSED_FILE"]["START_TIME"], meta["UNCOMPRESSED_FILE"]["STOP_TIME"])
            virtual_files.append(VirtualFile(file_dict["zip"], interval, meta))

        virtual_files.sort(key=lambda x: x.interval)
        remote_interval_list = IntervalList([file.interval for file in virtual_files])
        mask = remote_interval_list.intersection_mask(time_intervals)
        virtual_files = [vf for keep, vf in zip(mask, virtual_files) if keep]
        return virtual_files

    def _parse_current_file(self):
        # In case the current file is not downloaded yet, wait until it is
        self.current_file.wait_to_be_downloaded()
        # Assume self.current_file.file is fully loaded
        with zipfile.ZipFile(self.current_file.file, "r") as zip_file:
            # In DIVINER dataset, there is just a single data file within the ZIP archive
            with zip_file.open(zip_file.filelist[0]) as f:
                df = pd.read_csv(
                    io.StringIO(f.read().decode("utf-8")),
                    skiprows=3,
                    delimiter=",",
                    usecols=list(dtypes.keys()),
                    skipinitialspace=True,
                )

                def sclk_float_to_scs2e_str(arr: np.ndarray) -> List[str]:
                    sec = np.floor(arr).astype(int)
                    ticks = np.round((arr - sec) * 65536).astype(int)
                    return [f"{s}:{t}" for s, t in zip(sec, ticks)]

                sclk_strings = sclk_float_to_scs2e_str(df["sclk"].values)
                df["et"] = [spice.scs2e(LRO_INT_ID, sclk) for sclk in sclk_strings]
                df.sort_values("et", inplace=True)
                self.current_file.data = df

    def _get_interval_data_from_current_file(self, time_interval: TimeInterval):
        # We assume the file is parsed, as is implemented in _load_next_file method logic
        data = self.current_file.data.loc[
            (self.current_file.data.et > time_interval.start_et) & (self.current_file.data.et < time_interval.end_et)
        ].copy()
        data["c"] = -1
        data["det"] = -1
        data[["cx_dsk", "cy_dsk", "cz_dsk"]] = self.kernel_manager.dsk.latlon_to_cartesian(data["clat"], data["clon"])
        return data.to_dict(orient="records")

    def process_data_entry(
        self, data_entry: Dict, instrument: BaseInstrument, filter_obj: BaseFilter
    ) -> Optional[Dict]:
        et = data_entry["et"]
        self.kernel_manager.step_et(et)
        projection_vector = (
            instrument.sub_instruments[data_entry["c"]]
            .pixels[data_entry["det"]]
            .transformed_boresight(instrument.frame, et)
        )
        from src.simulation.simulator import Projection

        projection: Projection = instrument.project_vector(et, projection_vector)
        if filter_obj.point_pass(projection.projection):
            data_entry.update(projection.to_data())
            return data_entry
        else:
            return None
