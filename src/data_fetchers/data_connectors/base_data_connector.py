import logging
import requests
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from concurrent.futures import ThreadPoolExecutor, Future
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.SPICE.instruments.instrument import BaseInstrument
from src.filters import BaseFilter
from src.structures import VirtualFile
from src.structures import IntervalList, TimeInterval
from src.SPICE.kernel_utils.kernel_management import BaseKernelManager

logger = logging.getLogger(__name__)


class BaseDataConnector(ABC):
    """
    Base class for data connection, specific datasets are configured within their specific files.
    """

    name = None
    orbiting_body = "MOON"
    timeseries = None
    indices = None

    _session: Optional[requests.Session] = None
    _executor = ThreadPoolExecutor(max_workers=24)

    def __init__(
        self,
        time_intervals: IntervalList,
        kernel_manager: BaseKernelManager,
    ):
        """
        With config, we can specify:

        -  additional filtering conditions for virtual_files meta attribute (PDS3 parsed lbl file into dict)

        """
        self.kernel_manager = kernel_manager
        self.remote_files: List[VirtualFile] = self.discover_files(time_intervals)
        self.current_file_idx = 0
        # Start fetching the first file, start prefetching the next one too
        for file in self.remote_files[:2]:
            file.download()
        self.current_file.wait_to_be_downloaded()
        self._parse_current_file()
        if self.timeseries is None:
            raise NotImplementedError("Timeseries must be set in the derived class.")
        if self.indices is None:
            raise NotImplementedError("Indices must be set in the derived class.")

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
    def process_data_entry(self, data_entry: Dict, instrument: BaseInstrument, filter: BaseFilter) -> Dict:
        """
        Process a single data entry. Instrument will be reprojected and filtered with more precision

        If any other enhancements and attribute filtering is needed, it should be done here.
        """
        ...

    def validate_metadata(self, _: Dict) -> bool:
        """
        Validate the metadata of the file. This is a placeholder for any specific validation logic.
        """
        # Placeholder for actual validation logic
        return True

    def read_interval(self, time_interval: TimeInterval) -> List[Dict]:
        """
        Fetch data for the specified time interval. Files arefetched BASED on the interval list,
        so any discrepancies mean either file was not found or something terrible happened
        """
        data: List[Dict] = []

        # 1) Skip all files that end before our window starts
        while self.current_file.interval.is_interval_before(time_interval):
            if not self._load_next_file():
                logger.error("No files cover the start of the requested interval.")
                return []

        # 2) If the first remaining file starts after our window ends, nothing to do
        if self.current_file.interval.is_interval_after(time_interval):
            logger.error("First available file begins after the end of the requested interval.")
            return []

        # 3) Now consume every file that overlaps our window
        while True:
            file_iv = self.current_file.interval

            if not file_iv.overlaps(time_interval):
                break  # No more overlaps = done

            # Compute the exact slice of interest
            sub_iv = file_iv & time_interval
            if sub_iv is None:
                logger.warning(f"Unexpected: {file_iv} overlaps but & returned None. Falling back.")
                sub_iv = TimeInterval(
                    max(file_iv.start_et, time_interval.start_et),
                    min(file_iv.end_et, time_interval.end_et),
                )

            # Fetch the matching chunk from the current file
            chunk = self._get_interval_data_from_current_file(sub_iv)
            if not isinstance(chunk, list):
                raise TypeError(f"_get_interval_data_from_current_file must return List[Dict], got {type(chunk)}")

            data.extend(chunk)

            # Stop if we reached the end of the requested window
            if file_iv.end_et >= time_interval.end_et:
                break

            if not self._load_next_file():
                logger.error("Reached end of files before covering interval.")
                break
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
