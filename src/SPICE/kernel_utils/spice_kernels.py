"""
====================================================
SPICE Kernel Management Module
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

Overview:
---------
This module handles the downloading, caching, and timed management of SPICE kernels,
supporting both static and dynamic (time-bounded) kernels for lunar remote sensing simulations.

SPICE kernels have to be managed from the main process, otherwise ugly things could happen

Key Features:
-------------
1. **Download Management** – Robust sync/async downloading with retry logic and size checks.
2. **Kernel Caching** – Automatic on-disk storage with cleanup support.
3. **Time-Aware Loading** – Loads the appropriate kernel(s) for a given simulation timestamp.
4. **Metadata Parsing** – Extracts time intervals from `.LBL` metadata for dynamic kernels.
5. **Concurrent Operations** – Uses threading for parallel downloads and metadata processing.

Structure:
----------
- **BaseKernel** – Common functionality for downloading, checking, loading, and unloading kernels.
- **AutoUpdateKernel** – Selects the latest matching kernel from a remote folder.
- **LBLKernel** – Extends `BaseKernel` with support for an adjacent `.LBL` metadata file.
- **TimeBoundKernel** – Mixin class providing a time interval interface (`in_interval()`).
- **LBLDynamicKernel** – Combines LBL and time-bounded behavior; interval is parsed from metadata.
- **DynamicKernel** – Predefined time-bounded kernel with known interval.
- **StaticKernelLoader** – Downloads and loads static kernels (e.g., FK, IK).
- **DynamicKernelManager** – Manages a series of time-ordered dynamic kernels.
- **LBLDynamicKernelLoader** – Dynamic manager specialized for `.LBL`-based kernels.
- **LROCDynamicKernelLoader** – Extracts intervals from LROC kernel filenames.
- **KernelPriorityManager** – Manages fallback across multiple kernel managers.
"""

import os
import logging
import random
import re
import requests
import time
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from urllib.parse import urljoin
from typing import Optional, List, Dict, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from astropy.time import Time
import spiceypy as spice
from tqdm import tqdm
from bs4 import BeautifulSoup as bs

from src.SPICE.kernel_utils.locks import SharedFileUseLock
from filelock import FileLock
from src.global_config import TQDM_NCOLS, SUPRESS_TQDM
from src.SPICE.config import (
    MAX_LOADED_DYNAMIC_KERNELS,
    MAX_KERNEL_DOWNLOADS,
    MAX_RETRIES,
    SPICE_CHUNK_SIZE,
    SPICE_TOTAL_TIMEOUT,
    KERNEL_TIME_KEYS,
    SPICE_KERNEL_LOCK_DOWNLOAD_TIMEOUT,
    KERNEL_PREFETCH_COUNT,
    KERNEL_LOCK_POLL_INTERVAL,
)

logger = logging.getLogger(__name__)


#####################################################################################################################
#####                                            Single kernel utils                                            #####
#####################################################################################################################


class BaseKernel:
    """Base implementation of SPICE kernel"""

    url: str
    filename: str

    def __init__(self, url: str, filename: str, keep_kernel: bool = True) -> None:
        self._loaded = False
        self.url = url
        self.filename = filename
        self.corrupted = False
        self._keep_kernel = keep_kernel
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        # Obtain the lock for kernel and register the use
        # We do not really care whether the file exists now, but if it does, we ensure other process will not delete it
        self._lock = SharedFileUseLock(self.filename)
        self._lock.register_use()

    @property
    def file_exists(self) -> bool:
        """Check if the local kernel file exists"""
        return os.path.exists(self.filename)

    def ensure_downloaded(self) -> None:
        """Make sure the kernel is downloaded"""
        if not self.file_exists:
            self.download_file(self.url, self.filename)

    def unload(self) -> None:
        """Unload the kernel from spiceypy"""
        if self._loaded:
            spice.unload(self.filename)
            self._loaded = False
            logger.debug("Unloaded kernel %s", self.filename)
        else:
            logger.debug("Attempted to unload non-loaded kernel %s.", self.filename)
        # Release the lock and delete the file if not needed
        self._lock.release_use()
        if not self._keep_kernel:
            self.delete_file()

    def load(self) -> None:
        """Ensure the kernel is downloaded and load it into spiceypy."""
        if not self._loaded:
            self.ensure_downloaded()
            spice.furnsh(self.filename)
            self._loaded = True
            logger.debug("[SPICE-LOAD] Loading kernel: %s", self.filename)
        else:
            logger.debug("Kernel %s is already loaded", self.filename)

    def delete_file(self) -> None:
        """Delete the file from disk"""
        if self.file_exists:
            threading.Thread(target=lambda: self._lock.try_delete_file(), daemon=True).start()
            logger.debug("Scheduled async deletion of kernel file %s", self.filename)
        else:
            logger.debug("Attempted to delete non-existing kernel file %s", self.filename)

    def _verify_file_size(self, path: str, expected: Optional[int]) -> None:
        if expected is None:
            return
        actual = os.path.getsize(path)
        if actual != expected:
            raise ValueError(f"Size mismatch: expected {expected} bytes, got {actual} bytes")

    def download_file(self, url: str, filename: str) -> None:
        with FileLock(
            filename + ".tmp.lock", timeout=SPICE_KERNEL_LOCK_DOWNLOAD_TIMEOUT, poll_interval=KERNEL_LOCK_POLL_INTERVAL
        ):
            if not os.path.exists(filename):
                self._download_file(url, filename)

    ### Nesting crime scene start
    def _download_file(self, url: str, filename: str) -> None:
        """
        Synchronously download the file with resume support and retries.
        Includes file size verification using expected Content-Length.
        """
        retries = 0
        temp_path = filename + ".tmp"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyDownloader/1.0)"}
        corruption = False

        while retries < MAX_RETRIES:
            try:
                resume_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                if resume_pos:
                    headers["Range"] = f"bytes={resume_pos}-"

                with requests.get(url, stream=True, timeout=SPICE_TOTAL_TIMEOUT, headers=headers) as r:
                    if r.status_code == 416:  # Requested range not satisfiable
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        resume_pos = 0
                        headers.pop("Range", None)
                        continue  # Retry without resume

                    r.raise_for_status()
                    total_size = None

                    # Determine total expected size
                    if resume_pos:
                        cr = r.headers.get("Content-Range")
                        if cr and "/" in cr:
                            try:
                                total_size = int(cr.split("/")[-1])
                            except ValueError:
                                total_size = None
                        else:
                            logger.warning("Server did not return Content-Range; cannot resume. Restarting download.")
                            resume_pos = 0
                            headers.pop("Range", None)
                            continue  # Retry from beginning
                    else:
                        cl = r.headers.get("Content-Length")
                        total_size = int(cl) if cl and cl.isdigit() else None

                    if total_size is None or total_size <= 0:
                        if not corruption:
                            corruptio = True
                        else:
                            logger.warning(f"File {filename} looks corrupted on remote server {url}")
                            self.corrupted = True
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            return

                    mode = "ab" if resume_pos else "wb"
                    with open(temp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=SPICE_CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)

                # Verify the file size
                actual_size = os.path.getsize(temp_path)
                if total_size is not None and actual_size != total_size:
                    raise ValueError(f"Size mismatch: expected {total_size} bytes, got {actual_size} bytes")

                os.rename(temp_path, filename)
                logger.debug("Successfully downloaded %s", url)
                return

            except Exception as e:
                retries += 1
                logger.warning("Attempt %d failed to download %s: %s", retries, url, e)
                sleep_time = max((2**retries), 120) + (random.random() * retries)
                if retries < MAX_RETRIES:
                    time.sleep(sleep_time)
                else:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    logger.error("Failed to download %s after %s attempts", url, MAX_RETRIES)
                    return

    def __repr__(self):
        return f"<BaseKernel filename='{self.filename}'>"


class AutoUpdateKernel(BaseKernel):
    """Kernel that automatically picks the newest file from a remote folder, based on kernel name, assuming datetime is present."""

    url: str
    filename: str

    def __init__(self, folder_url: str, folder_path: str, regex: str, keep_kernel: bool = True):
        self.regex = re.compile(regex)
        response = requests.get(folder_url)
        response.raise_for_status()
        soup = bs(response.text, "html.parser")
        links = soup.find_all("a")
        matches = [link.get("href") for link in links if self.regex.search(link.get("href") or "")]
        if not matches:
            raise ValueError(f"No matching kernel found in {folder_url} with regex {regex}")
        filename = sorted(matches, reverse=True)[0]
        url = urljoin(folder_url, filename)
        super().__init__(url, os.path.join(folder_path, filename), keep_kernel=keep_kernel)


class LBLKernel(BaseKernel):

    def __init__(
        self, url: str, filename: str, metadata_url: str, metadata_filename: str, keep_kernel: bool = True
    ) -> None:
        BaseKernel.__init__(self, url, filename, keep_kernel=keep_kernel)
        self.metadata_url = metadata_url
        self.metadata_filename = metadata_filename

    @property
    def metadata_exists(self) -> bool:
        """Check if the metadata file exists on disk"""
        return os.path.exists(self.metadata_filename)

    def download_metadata(self) -> None:
        """Download the metadata file from the internet"""
        if not self.metadata_exists:
            logger.debug("Downloading metadata: %s", self.metadata_filename)
            self.download_file(self.metadata_url, self.metadata_filename)
        else:
            logger.debug("Metadata already exists: %s", self.metadata_filename)

    def delete_metadata(self) -> None:
        if os.path.exists(self.metadata_filename):
            os.remove(self.metadata_filename)
            logger.debug("Deleted metadata %s", self.metadata_filename)
        else:
            logger.debug("Attempted to delete non-existing metadata %s", self.metadata_filename)


class StaticKernelLoader:
    """Downloads and loads a static group of SPICE kernels from disk or remote."""

    def __init__(self, kernel_objects: OrderedDict[str, List[BaseKernel]]):
        # We want to aggregate all kernels in one list while retaining its order
        self.kernel_pool = [kernel for kernels in kernel_objects.values() for kernel in kernels]
        pbar = tqdm(
            total=len(self.kernel_pool), desc="Downloading static kernels", ncols=TQDM_NCOLS, disable=SUPRESS_TQDM
        )

        def _download_kernel(kernel: BaseKernel, pbar: tqdm) -> None:
            kernel.ensure_downloaded()
            pbar.update(1)

        with ThreadPoolExecutor(max_workers=MAX_KERNEL_DOWNLOADS) as executor:
            futures = [executor.submit(_download_kernel, kernel, pbar) for kernel in self.kernel_pool]
            for future in as_completed(futures):
                # Optionally: future.result() to raise exceptions
                pass

        pbar.close()

    def load(self):
        logger.debug("Loading %d static kernels...", len(self.kernel_pool))
        if not self.kernel_pool:
            logger.warning("No kernels to load.")
        for kernel in self.kernel_pool:
            kernel.load()

    def unload(self):
        logger.debug("Unloading %d static kernels...", len(self.kernel_pool))
        if not self.kernel_pool:
            logger.warning("No kernels to unload.")
        for kernel in self.kernel_pool:
            kernel.unload()


#####################################################################################################################
#####                                            Dynamic kernel utils                                           #####
#####################################################################################################################


class TimeBoundKernel(BaseKernel):
    """
    Kernel with a defined valid time interval.

    Used as a base for kernels that are only applicable within a specific time range.
    Is inten
    """

    time_start: Optional[Time]
    time_stop: Optional[Time]

    def __init__(
        self, url: str, filename: str, time_start: Optional[Time], time_stop: Optional[Time], keep_kernel: bool = True
    ):
        """
        Initialize a time-bounded kernel.

        Parameters:
        - url: Remote URL of the kernel file.
        - filename: Local path where the kernel should be saved.
        - time_start: Start of the valid time interval.
        - time_stop: End of the valid time interval.
        """
        super().__init__(url, filename, keep_kernel=keep_kernel)
        self.time_start = time_start
        self.time_stop = time_stop

    def in_interval(self, time: Time) -> bool:
        """
        Check if a given time is within this kernel's valid interval. Fails if start and stop times not set

        Parameters:
        - time: Time to check.

        Returns:
        - True if time is within interval, False otherwise.
        """
        return self.time_start <= time <= self.time_stop


class LBLDynamicKernel(LBLKernel, TimeBoundKernel):
    """
    Dynamic kernel with .LBL metadata file specifying time interval.
    """

    def __init__(self, url, filename, metadata_url, metadata_filename, keep_kernel: bool = True):
        """
        Initialize an LBL-based dynamic kernel.

        Parameters:
        - url: Kernel file URL.
        - filename: Local kernel file path.
        - metadata_url: Metadata (.LBL) file URL.
        - metadata_filename: Local metadata file path.
        """
        LBLKernel.__init__(self, url, filename, metadata_url, metadata_filename, keep_kernel=keep_kernel)
        self.time_start = None
        self.time_stop = None

    def get_time_interval(self, pbar: tqdm) -> bool:
        """
        Download and parse the .LBL metadata to extract the kernel's valid time interval.

        Parameters:
        - pbar: Progress bar to update after completion.

        Returns:
        - True if time interval was successfully parsed, False otherwise.
        """
        self.download_metadata()

        pattern = re.compile(r"(\S+)\s*=\s*(.+)")
        with open(self.metadata_filename, "r") as f:
            content = f.read()

        metadata = {m.group(1): m.group(2).strip('"') for m in pattern.finditer(content)}
        if not all(key in metadata for key in KERNEL_TIME_KEYS.values()):
            logger.warning("Skipping %s, insufficient data", self.metadata_filename)
            return False

        self.time_start = Time(metadata[KERNEL_TIME_KEYS["time_start_key"]], format="isot", scale="utc")
        self.time_stop = Time(metadata[KERNEL_TIME_KEYS["time_stop_key"]], format="isot", scale="utc")
        if pbar:
            pbar.update(1)
        return True


class DynamicKernel(TimeBoundKernel):
    """
    Simple time-bounded kernel initialized with pre-known interval.
    """

    def __init__(self, url: str, filename: str, time_start: Time, time_stop: Time, keep_kernel: bool = True):
        """
        Initialize a kernel with known time interval.

        Parameters:
        - url: Remote kernel file URL.
        - filename: Local storage path.
        - time_start: Start of the valid time interval.
        - time_stop: End of the valid time interval.
        """
        super().__init__(url, filename, time_start, time_stop, keep_kernel=keep_kernel)


class DynamicKernelManager(ABC):

    @property
    def min_loaded_time(self) -> Time:
        if not self.kernel_pool:
            return None
        return self.kernel_pool[0].time_start

    @property
    def max_loaded_time(self) -> Time:
        if not self.kernel_pool:
            return None
        return self.kernel_pool[-1].time_stop

    @abstractmethod
    def parse_kernel_time_bounds(self, filename, url) -> Dict:
        """
        Get the kernel pool kwargs for the given filename and url
        """
        ...

    def __init__(
        self,
        path: str,
        base_url: str,
        regex: str,
        keep_kernels: bool = True,
        pre_download_kernels: bool = True,
        prefetch_kernels: int = KERNEL_PREFETCH_COUNT,
        min_time_to_load: Optional[Time] = None,
        max_time_to_load: Optional[Time] = None,
        time_intervals: Optional[Tuple[Time, Time]] = None,
    ):
        """
        path: local path to dynamic kernel folder
        base_url: remote location of dynamic kernels
        regex: filter all files from local path to get desired kernels
        keep_kernels: if True, downloaded files are stored on disk. If false, once data are unloaded,
                    corresponding files are deleted
        pre_download_kernels: if True, all kernels are downloaded at initialization (if not already on the disk)
        frefetch_kernels: number of kernels to asynchronously prefetch
        min_time_to_load: minimum time to load the kernel, ditch kernels with max_time lower then this value
        max_time_to_load: maximum time to load the kernel, ditch kernels with min_time higher then this value
        time_intervals: Optional[Tuple[Time, Time]] = None - This is another way of contraining the required time to load (sorted)
        """
        self.regex = re.compile(regex)
        self.base_url = base_url
        self.keep_kernels = keep_kernels
        self.pre_download_kernels = pre_download_kernels
        self.prefetch_kernels = prefetch_kernels
        self.path = path
        self.time_intervals = time_intervals
        self.min_time_to_load = min_time_to_load
        self.max_time_to_load = max_time_to_load
        # self.load_callbacks = load_callbacks
        os.makedirs(self.path, exist_ok=True)

        self.active_kernel_id = -1
        self.loaded_kernels = []
        self.load_metadata()

        if self.pre_download_kernels:
            self.download_kernels()
        elif self.prefetch_kernels > 0:
            for kernel in self.kernel_pool[: self.prefetch_kernels]:
                # Fire and forget thread to download data, concurrency is solved with file locks already
                threading.Thread(target=kernel.ensure_downloaded, daemon=True).start()

    def apply_time_interval_kernel_filter(self) -> None:
        """
        Filters the kernel pool based on:
        - min_time_to_load
        - max_time_to_load
        - time_intervals (list of (start, end) Time objects)
        """

        def kernel_is_valid(kernel):
            # Min/max time checks
            if self.min_time_to_load and kernel.time_stop <= self.min_time_to_load:
                return False
            if self.max_time_to_load and kernel.time_start >= self.max_time_to_load:
                return False

            # Time intervals check (if defined)
            if self.time_intervals:
                # Keep if any interval overlaps
                for interval_start, interval_end in self.time_intervals:
                    if kernel.time_stop > interval_start and kernel.time_start < interval_end:
                        return True
                return False  # No overlaps
            return True  # Passed all checks

        new_pool = []
        for kernel in self.kernel_pool:
            if kernel_is_valid(kernel):
                new_pool.append(kernel)
            else:
                kernel.unload()
        self.kernel_pool = new_pool

    def load_metadata(self):
        response = requests.get(self.base_url)
        soup = bs(response.text, "html.parser")
        links = soup.find_all("a")

        kernel_filenames = [link.get("href") for link in links if self.regex.search(link.get("href") or "")]
        kernel_urls = [urljoin(self.base_url, filename) for filename in kernel_filenames]

        self.kernel_pool: List[DynamicKernel] = [
            DynamicKernel(
                kernel_url,
                os.path.join(self.path, kernel_filename),
                keep_kernel=self.keep_kernels,
                **self.parse_kernel_time_bounds(kernel_filename, kernel_url),
            )
            for kernel_url, kernel_filename in zip(kernel_urls, kernel_filenames)
        ]
        self.apply_time_interval_kernel_filter()
        self.kernel_pool_len = len(self.kernel_pool)
        self.kernel_pool = sorted(self.kernel_pool, key=lambda x: x.time_start)

    def download_kernels(self) -> None:
        """
        Downloads all kernels that are not already on disk
        """

        def download_kernel(kernel: LBLDynamicKernel, pbar: tqdm) -> None:
            kernel.ensure_downloaded()
            pbar.update(1)

        pbar = tqdm(
            total=len(self.kernel_pool), desc="Downloading dynamic kernels", ncols=TQDM_NCOLS, disable=SUPRESS_TQDM
        )

        with ThreadPoolExecutor(max_workers=MAX_KERNEL_DOWNLOADS) as executor:
            futures = [executor.submit(download_kernel, kernel, pbar) for kernel in self.kernel_pool]
            for future in as_completed(futures):
                future.result()

        pbar.close()

    def load_new_kernel(self, _id: int) -> None:
        """
        Load a new kernel to spiceypy with all needed actions

        _id points to kernel list
        """
        # Fire and forget thread to download data, concurrency is solved with file locks already
        if self.prefetch_kernels > 0 and not self.pre_download_kernels:
            for kernel in self.kernel_pool[_id + 1 : _id + self.prefetch_kernels]:
                threading.Thread(target=kernel.ensure_downloaded, daemon=True).start()

        kernel = self.kernel_pool[_id]
        self.active_kernel_id = _id
        kernel.load()

        self.loaded_kernels.append(kernel)
        if len(self.loaded_kernels) > MAX_LOADED_DYNAMIC_KERNELS:
            old_kernel = self.loaded_kernels.pop(0)
            logger.debug("Unloading kernel due to pool limit: %s", old_kernel.filename)
            old_kernel.unload()

    def reload_kernels(self, time: Time) -> bool:
        """
        Reloads kernels for given time. Return True if a kernel for requested Time is already loaded or found and loaded
        """
        for loaded_kernel in self.loaded_kernels[::-1]:
            if loaded_kernel.in_interval(time):
                return True

        if self.active_kernel_id < self.kernel_pool_len - 1 and self.kernel_pool[self.active_kernel_id + 1].in_interval(
            time
        ):
            self.load_new_kernel(self.active_kernel_id + 1)
            return True
        else:
            for i, kernel in enumerate(self.kernel_pool):
                if kernel.in_interval(time):
                    self.load_new_kernel(i)
                    return True
        return False

    def unload(self) -> None:
        """
        Unload all loaded kernels
        """
        for kernel in self.loaded_kernels:
            kernel.unload()
        self.loaded_kernels = []
        self.active_kernel_id = -1
        self.kernel_pool = []


class LBLDynamicKernelLoader(DynamicKernelManager):

    def __init__(
        self,
        path: str,
        base_url: str,
        regex: str,
        metadata_regex: str,
        keep_kernels: bool = True,
        pre_download_kernels: bool = True,
        prefetch_kernels: int = KERNEL_PREFETCH_COUNT,
        min_time_to_load: Optional[Time] = None,
        max_time_to_load: Optional[Time] = None,
        time_intervals: Optional[Tuple[Time, Time]] = None,
    ):
        """
        path: local path to dynamic kernel folder
        base_url: remote location of dynamic kernels
        regex: filter all files from local path to get desired kernels
        metadata_regex: filter all files with metadata, implicitly .lbl files
        keep_kernels: if True, downloaded files are stored on disk. If false, once data are unloaded,
                    corresponding files are deleted
        pre_download_kernels: if True, all kernels are downloaded at initialization (if not already on the disk)
        load_callbakcs: list of tuples with callback functions, their arguments and keyword arguments. Checks for correct
            loading of kernels. Could throw exception once to load it properly, throwing 2 means something went wrong.
            First element of the iterable is reserved for et (placeholder is needed when defining the callbacks)
            (commented out for now, but in case it would be needed, uncommenting the code will bring it back to life)
        time_intervals: Optional[Tuple[Time, Time]] = None - This is another way of contraining the required time to load (sorted)
        """
        self.metadata_regex = re.compile(metadata_regex)
        super().__init__(
            path,
            base_url,
            regex,
            keep_kernels,
            pre_download_kernels,
            prefetch_kernels,
            min_time_to_load,
            max_time_to_load,
            time_intervals,
        )

    def parse_kernel_time_bounds(self, filename, url) -> Dict:
        """
        Is not being used, have to be overwritten, because is abstract in the parent class
        """
        ...

    def load_metadata(self) -> List[LBLDynamicKernel]:
        """
        Creates a list of dynamic kernels by crawling the base_url
        Reads their time intervals
        """
        response = requests.get(self.base_url)
        soup = bs(response.text, "html.parser")
        links = soup.find_all("a")

        kernel_filenames = [link.get("href") for link in links if self.regex.search(link.get("href") or "")]
        metadata_filenames = [link.get("href") for link in links if self.metadata_regex.match(link.get("href") or "")]
        kernel_urls = [urljoin(self.base_url, filename) for filename in kernel_filenames]
        metadata_urls = [urljoin(self.base_url, filename) for filename in metadata_filenames]

        self.kernel_pool: List[LBLDynamicKernel] = [
            LBLDynamicKernel(
                kernel_url,
                os.path.join(self.path, kernel_filename),
                metadata_url,
                os.path.join(self.path, metadata_filename),
                keep_kernel=self.keep_kernels,
            )
            for kernel_url, kernel_filename, metadata_url, metadata_filename in zip(
                kernel_urls, kernel_filenames, metadata_urls, metadata_filenames
            )
        ]

        pbar = tqdm(
            total=len(self.kernel_pool),
            desc="Downloading dynamic kernel metadata",
            ncols=TQDM_NCOLS,
            disable=SUPRESS_TQDM,
        )

        with ThreadPoolExecutor(max_workers=MAX_KERNEL_DOWNLOADS) as executor:
            futures = [executor.submit(kernel.get_time_interval, pbar) for kernel in self.kernel_pool]
            for future in as_completed(futures):
                future.result()

        pbar.close()

        self.apply_time_interval_kernel_filter()
        self.kernel_pool_len = len(self.kernel_pool)
        self.kernel_pool = sorted(self.kernel_pool, key=lambda x: x.time_start)


class LROCDynamicKernelLoader(DynamicKernelManager):
    FILENAME_PATTERN = re.compile(r"lrolc_(\d{7})_(\d{7})")

    def parse_kernel_time_bounds(self, filename: str, url: str) -> Dict[str, Time]:
        """
        Extracts time_start and time_stop from filenames matching the LROC kernel pattern.
        Expected format: lrolc_YYYYDDD_YYYYDDD.*
        """
        match = self.FILENAME_PATTERN.search(filename)
        if not match:
            raise ValueError(f"Filename does not match expected LROC pattern: {filename}")

        start_doy, end_doy = match.groups()
        start_str = f"{start_doy[:4]}:{start_doy[4:]}"
        end_str = f"{end_doy[:4]}:{end_doy[4:]}"
        t_start = Time(start_str, format="yday", scale="utc")
        t_end = Time(end_str, format="yday", scale="utc")

        logger.debug("Parsed LROC kernel time range from '%s': %s → %s", filename, t_start.iso, t_end.iso)
        return {"time_start": t_start, "time_stop": t_end}


class PriorityKernelLoader:
    """
    Manages multiple kernel loaders with priority order.
    Ensures at least one kernel covers the given time.
    """

    def __init__(self, kernel_managers: Sequence[LBLDynamicKernelLoader]) -> None:
        self.kernel_managers = kernel_managers

    @property
    def min_loaded_time(self) -> Optional[Time]:
        """Earliest time covered by any loader"""
        if not self.kernel_managers:
            return None

        return min(km.min_loaded_time for km in self.kernel_managers if km.min_loaded_time is not None)

    @property
    def max_loaded_time(self) -> Optional[Time]:
        """Latest time covered by any loader"""
        if not self.kernel_managers:
            return None
        return max(km.max_loaded_time for km in self.kernel_managers if km.max_loaded_time is not None)

    def reload_kernels(self, time: Time) -> bool:
        """
        Attempt to reload kernels for the given time.
        Returns True if any loader handled it.
        """
        for i, km in enumerate(self.kernel_managers):
            if km.reload_kernels(time):
                logger.debug("Time %s handled by kernel manager #%d", time.iso, i)
                return True
        logger.warning("No kernel manager could handle time %s", time.iso)
        return False

    def unload(self) -> None:
        """Unload all kernels from all managers"""
        for km in self.kernel_managers:
            km.unload()
