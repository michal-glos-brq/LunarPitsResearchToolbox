import logging
import requests
from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from astropy.time import Time
from concurrent.futures import ThreadPoolExecutor, Future
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from cachetools import LRUCache

from src.SPICE.instruments.instrument import BaseInstrument
from src.simulation.filters import BaseFilter
from src.data_fetchers.data_connectors.virtual_file import VirtualFile
from src.data_fetchers.interval_manager import IntervalList, TimeInterval


logger = logging.getLogger(__name__)


class BaseConnectorConfig:
    # Pass a list of callables. When callable returns False, the file is filtered out
    # Usefull for data sw version, dataquaolity flags and similar
    virtual_file_meta_filters: List[Callable] = []

    def filter_virtual_file_metadata(self, virtual_file: VirtualFile) -> bool:
        """Returns True if success, false if to be filtered out"""
        return all(filter_func(virtual_file.metadata) for filter_func in self.virtual_file_meta_filters)


class BaseDataConnector(ABC):
    """
    Base class for data connection, specific datasets are configured within their specific files.
    """

    _session: Optional[requests.Session] = None
    _executor = ThreadPoolExecutor(max_workers=24)

    def __init__(self, time_intervals: IntervalList, config: Optional[BaseConnectorConfig] = None):
        """
        With config, we can specify:

        -  additional filtering conditions for virtual_files meta attribute (PDS3 parsed lbl file into dict)

        """
        self.config = config or BaseConnectorConfig()
        self.remote_files: List[VirtualFile] = self.discover_files(time_intervals)
        self.current_file_idx = 0
        # Start fetching the first file, start prefetching the next one too
        for file in self.remote_files[:2]:
            file.download()
        self.current_file.wait_to_be_downloaded()
        self._parse_current_file()

    # Download logic implemented below is meant for smaller files
    @classmethod
    def _init_session(cls):
        if cls._session is None:
            retry = Retry(
                total=24,
                backoff_factor=0.5,
                status_forcelist=(403, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET"]),
                raise_on_status=True,
            )
            adapter = HTTPAdapter(max_retries=retry, pool_connections=512, pool_maxsize=512)  # Well above max workers
            s = requests.Session()
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            cls._session = s

    @classmethod
    def fetch_url(cls, url: str, timeout: int = 60) -> requests.Response:
        """
        Synchronous fetch with retry logic.
        Raises on HTTP errors.
        """
        cls._init_session()
        resp = cls._session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp

    @classmethod
    def fetch_url_async(cls, url: str, timeout: int = 60) -> Future:
        """
        Returns a Future that will resolve to a requests.Response.
        """
        return cls._executor.submit(cls.fetch_url, url, timeout)

    @property
    def current_file(self) -> Optional[VirtualFile]:
        """Return the current file being processed."""
        return self.remote_files[self.current_file_idx]

    @abstractmethod
    def discover_files(self, time_intervals: IntervalList) -> List[VirtualFile]:
        """Discover files to be processed. Take the interval into account"""
        ...

    @abstractmethod
    def _get_interval_data_from_current_file(self, time_interval: IntervalList) -> List[Dict]:
        """Fetch data of the given interval from currently loaded dataframe or other data format"""
        ...

    @abstractmethod
    def _parse_current_file(self):
        """Parse the current file. From BytesIO to proferrable DataFrame"""
        ...

    @abstractmethod
    def process_data_entry(
        self, data_entry: Dict, instrument: BaseInstrument, filter: BaseFilter, scpos_cache: Optional[LRUCache] = None
    ) -> Dict:
        """
        Process a single data entry. Instrument will be reprojected and filtered with more precision

        If any other enhancements and attribute filtering is needed, it should be done here.
        """
        ...

    def read_interval(self, time_interval: TimeInterval) -> List[Dict]:
        """Fetch data for the specified time interval. Files arefetched BASED on the interval list, so any discrepancies mean either file was not found or something terrible happened"""
        while not self.current_file.interval.overlaps(
            time_interval
        ) and not self.current_file.interval.is_interval_after(time_interval):
            # If the current file does not overlap with the requested time interval, load the next file
            if not self._load_next_file():
                logger.error("No more files to load, something went wrong")
                return []

        if not self.current_file.interval.overlaps(time_interval):
            raise ValueError(f"Inconsistent timeinterval: {time_interval} not in current file")

        data = []
        while self.current_file.interval.overlaps(time_interval) and not self.current_file.interval.is_subinterval(
            time_interval
        ):
            # This means the requested interval overlaps with teh current file intererval, but oges further
            data += self._get_interval_data_from_current_file(time_interval)
            if not self._load_next_file():
                logger.error("No more files to load, something went wrong")
                return data

        if self.current_file.interval.overlaps(time_interval):
            data += self._get_interval_data_from_current_file(time_interval)

        return data

    def _load_next_file(self) -> bool:
        """Update the state of the object to point to the next file."""
        self.current_file.unload()
        if self.current_file_idx + 1 < len(self.remote_files):
            self.current_file_idx += 1
            # If more files to be downloaded, start downloading the next one
            if self.current_file_idx + 1 < len(self.remote_files):
                self.remote_files[self.current_file_idx + 1].download()

            self._parse_current_file()
            return True
        return False
