"""
WARNING: When assigning tasks, we are looking for separate day datasets, but there are some 0.1s overlaps. If you
do not want to risk data loss by omission, assign the tasks with times in the middle of the day
"""

import zipfile
import pvl
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
from bs4 import BeautifulSoup as bs
import spiceypy as spice
from astropy.time import Time, TimeDelta

from src.structures import VirtualFile
from src.structures import TimeInterval, IntervalList
from src.data_fetchers.data_connectors.base_data_connector import BaseDataConnector
from src.data_fetchers.config import DIVINER_BASE_URL, DIVINER_YEARLY_DATA_URLS
from src.SPICE.config import LRO_INT_ID
from src.SPICE.utils import DatetimeToETDecoder
from src.global_config import SUPRESS_TQDM, TQDM_NCOLS
from src.filters import BaseFilter
from src.SPICE.instruments.instrument import BaseInstrument
from src.structures import ProjectionPoint


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


compose_diviner_url = lambda path: urljoin(DIVINER_BASE_URL, path)

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

astropy_time_units = ["year", "month", "day", "hour", "minute", "second"]


class DivinerDataConnector(BaseDataConnector):

    name = "DIVINER"

    timeseries = {
        "timeField": "timestamp",
        "metaField": "meta",
        "granularity": "seconds",
    }

    indices = [
        "meta.et",  # always index your time field
        "orbit",  # if you ever query by orbit
        "c",  # channel
        "det",  # detector
        "cloctime",  # Local time (helps with diurnal effects)
        "csunazi",  # Solar azimuth
        "csunzen",  # Solar zenith
        "cemis",  # Emission angle
        "cx_projected",  # Projected coordinates
        "cy_projected",
        "cz_projected",
        "meta.extraction_name",  # if you ever query by simulation name
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
            url = urlparse(DIVINER_YEARLY_DATA_URLS[current_year])
            tasks.append((current_year, url, self.fetch_url_async(url.geturl())))

        month_urls: Dict[Tuple[int, int], str] = {}

        for year, url, task in tasks:
            response = task.result()
            soup = bs(response.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and href.startswith(url.path):
                    month_urls[(year, int(href[-3:-1]))] = compose_diviner_url(href)

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

        for year, month, url, task in tqdm(
            tasks, desc="DIVINER exploring Yearly URLs", disable=SUPRESS_TQDM, ncols=TQDM_NCOLS
        ):
            response = task.result()
            soup = bs(response.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and href.startswith(url.path):
                    daily_urls[(year, month, int(href[-3:-1]))] = compose_diviner_url(href)

        # Drop the days that are not in the time intervals
        for year, month, day in tqdm(
            list(daily_urls.keys()), desc="DIVINER fetching Yearly URLs", disable=SUPRESS_TQDM, ncols=TQDM_NCOLS
        ):
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
        for (year, month, day), url in tqdm(daily_urls.items(), desc="DIVINER exploring Daily URLs", disable=SUPRESS_TQDM, ncols=TQDM_NCOLS):
            url = urlparse(url)
            tasks.append((year, month, day, url, self.fetch_url_async(url.geturl())))

        minute_urls: Dict[Tuple[int, int, int, int, int], str] = defaultdict(dict)

        for year, month, day, url, task in tqdm(tasks, desc="DIVINER fetching Daily URLs", disable=SUPRESS_TQDM, ncols=TQDM_NCOLS):
            response = task.result()
            soup = bs(response.text, "html.parser")
            for link in soup.find_all("a"):
                href = link.get("href")
                if href and href.startswith(url.path):
                    # Parse hours and minutes too, use file suffix as key to the other dict
                    minute_urls[(year, month, day, int(href[-12:-10]), int(href[-10:-8]))][href[-3:].lower()] = (
                        compose_diviner_url(href)
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

        # We obtain all the minute urls. Not to torture NAIF servers, we filter it further (with minute of tolerance to be inclusive)
        minute_urls_of_interest = {}
        for key, value in tqdm(list(minute_urls.items()), desc="DIVINER filtering minute URLs", disable=SUPRESS_TQDM, ncols=TQDM_NCOLS):
            base_datefile_time_dict = {time_key: time_value for time_key, time_value in zip(astropy_time_units, key)}
            base_datafile_timestamp = Time(base_datefile_time_dict, format="ymdhms", scale="utc")
            base_datefile_timestamp_et = spice.utc2et(base_datafile_timestamp.iso)
            # Subtract 60 seconds to be inclusive, will be enhanced with .lbl file data
            base_datefile_timestamp_et_start = base_datefile_timestamp_et - 60
            # Interval is expected to last 10 minutes, we add next 60 seconds to be strictly inclusive
            base_datefile_timestamp_et_end = base_datefile_timestamp_et + 660
            inclusive_time_interval = TimeInterval(base_datefile_timestamp_et_start, base_datefile_timestamp_et_end)
            if time_intervals.get_intervals_intersection(inclusive_time_interval):
                minute_urls_of_interest[key] = value

        lbl_tasks: List[Tuple[Dict[str, str], Future]] = []
        for file_dict in tqdm(list(minute_urls_of_interest.values()), desc="DIVINER exploring lbl metadata", disable=SUPRESS_TQDM, ncols=TQDM_NCOLS):
            lbl_tasks.append((file_dict, self.fetch_url_async(file_dict["lbl"])))

        virtual_files: List[VirtualFile] = []
        for file_dict, fut in tqdm(lbl_tasks, desc="Downloading DIVINER LBL metadata", disable=SUPRESS_TQDM, ncols=TQDM_NCOLS):
            resp = fut.result()
            meta = pvl.loads(resp.text, decoder=DatetimeToETDecoder())
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

    def _get_interval_data_from_current_file(self, time_interval: TimeInterval, _: BaseInstrument, __: BaseFilter) -> List[Dict]:
        # We assume the file is parsed, as is implemented in _load_next_file method logic
        data = self.current_file.data.loc[
            (self.current_file.data.et > time_interval.start_et) & (self.current_file.data.et < time_interval.end_et)
        ].copy()
        data["c"] += -1
        data["det"] += -1
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

        projection: ProjectionPoint = instrument.project_vector(et, projection_vector)
        if filter_obj.point_pass(projection.projection):
            data_entry.update(projection.to_data())
            return data_entry
        else:
            return None
