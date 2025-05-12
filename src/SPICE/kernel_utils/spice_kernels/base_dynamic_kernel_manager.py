"""
====================================================
SPICE Kernel Management Module
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

import os
import logging
import re
import requests
import threading
from abc import ABC, abstractmethod
from urllib.parse import urljoin
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from astropy.time import Time
from tqdm import tqdm
from bs4 import BeautifulSoup as bs


from src.global_config import TQDM_NCOLS, SUPRESS_TQDM
from src.SPICE.config import (
    MAX_LOADED_DYNAMIC_KERNELS,
    MAX_KERNEL_DOWNLOADS,
    KERNEL_PREFETCH_COUNT,
)
from src.SPICE.kernel_utils.spice_kernels.dynamic_kernels import DynamicKernel, LBLDynamicKernel

logger = logging.getLogger(__name__)


class DynamicKernelManager(ABC):
    """
    This is base class for Dynamic Kernel managers - an object responsible for orchestrating SPICE kernels in temporal dimension.
    """

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
        Get the kernel pool kwargs (DyanmicKernel constructor **kwargs) for the given filename and url.
        Used predominantly to load the time interval of the kernel.
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
        Filter out dynamic kernels from the pool that do not overlap with the desired time window.

        Kernels are validated against:
            - `min_time_to_load`: discard kernels ending before this time.
            - `max_time_to_load`: discard kernels starting after this time.
            - `time_intervals`: a list of (start, end) Time tuples; only keep kernels that overlap any of them.

        Kernels failing all criteria are unloaded and removed from the pool.
        """

        def kernel_is_valid(kernel) -> bool:
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

    def load_metadata(self) -> None:
        """
        Crawl the remote kernel directory, match kernel filenames using the configured regex,
        and parse their time bounds using `parse_kernel_time_bounds`.

        Constructs a list of `DynamicKernel` instances and filters them based on time constraints.
        The resulting `kernel_pool` is sorted by start time for efficient future access.
        """
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
        Downloads all kernels from kernel pool, that are not already on downloaded.
        """

        def download_kernel(kernel: LBLDynamicKernel, pbar: tqdm) -> None:
            kernel.ensure_downloaded()
            pbar.update(1)

        try:
            kernel_type = self.filename.rsplit("/", 2)[1]
        except Exception as e:
            kernel_type = "SPICE"

        pbar = tqdm(
            total=len(self.kernel_pool),
            desc=f"Downloading ({kernel_type}) dynamic kernels ",
            ncols=TQDM_NCOLS,
            disable=SUPRESS_TQDM,
        )

        with ThreadPoolExecutor(max_workers=MAX_KERNEL_DOWNLOADS) as executor:
            futures = [executor.submit(download_kernel, kernel, pbar) for kernel in self.kernel_pool]
            for future in as_completed(futures):
                future.result()

        pbar.close()

    def load_new_kernel(self, _id: int) -> None:
        """
        Loads the kernel with furnsh (_id is the index of the kernel in the kernel pool)

        If there are too many kernels loaded, the oldest one is unloaded.
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
        Ensure that a kernel covering the specified time is loaded.

        Checks if the time is already covered by any loaded kernel.
        If not, it searches the kernel pool for a matching interval, loads the kernel if found,
        and unloads the oldest if exceeding the kernel limit.

        Parameters:
            time (Time): The observation or simulation time requiring coverage.

        Returns:
            bool: True if a kernel covering `time` is now loaded, False otherwise.
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
