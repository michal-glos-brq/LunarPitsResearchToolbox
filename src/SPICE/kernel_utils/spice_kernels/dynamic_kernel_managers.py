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
from urllib.parse import urljoin
from typing import Optional, List, Dict, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from astropy.time import Time
from tqdm import tqdm
from bs4 import BeautifulSoup as bs


from src.global_config import TQDM_NCOLS, SUPRESS_TQDM
from src.SPICE.config import (
    MAX_KERNEL_DOWNLOADS,
    KERNEL_PREFETCH_COUNT,
)
from src.SPICE.kernel_utils.spice_kernels.dynamic_kernels import LBLDynamicKernel
from src.SPICE.kernel_utils.spice_kernels.base_dynamic_kernel_manager import DynamicKernelManager


logger = logging.getLogger(__name__)


class LBLDynamicKernelLoader(DynamicKernelManager):
    """
    Dynamic kernel loader for LBL (labelled) SPICE kernels.

    This subclass parses metadata files (e.g., .LBL) alongside kernel files
    to extract valid time intervals. It supports filtering, downloading,
    and managing a pool of LBLDynamicKernel instances.

    Metadata and kernel filenames are filtered using separate regex patterns.
    """

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
        Initialize the LBLDynamicKernelLoader.

        Parameters:
            path (str): Local directory where kernels and metadata will be stored.
            base_url (str): Base URL of the remote kernel directory.
            regex (str): Regex to match valid kernel filenames.
            metadata_regex (str): Regex to match valid metadata (.lbl) filenames.
            keep_kernels (bool): If False, downloaded files will be deleted after unloading.
            pre_download_kernels (bool): If True, kernels will be downloaded at initialization.
            prefetch_kernels (int): Number of kernels to prefetch asynchronously.
            min_time_to_load (Optional[Time]): Minimum time to retain kernels.
            max_time_to_load (Optional[Time]): Maximum time to retain kernels.
            time_intervals (Optional[Tuple[Time, Time]]): List of explicit time intervals to keep.
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
        Required abstract method (unused).

        Not needed because time bounds are parsed from LBL metadata
        during `load_metadata()`.
        """
        ...

    def load_metadata(self) -> List[LBLDynamicKernel]:
        """
        Crawl remote directory and initialize LBLDynamicKernel instances.

        For each kernel-metadata pair:
            - Download metadata (if needed).
            - Parse time interval via `get_time_interval()`.
            - Filter by time constraints.
            - Populate and sort the kernel pool.

        Returns:
            List[LBLDynamicKernel]: Loaded and filtered kernel list.
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
                os.path.join(self.path, os.path.basename(kernel_filename)),
                metadata_url,
                os.path.join(self.path, os.path.basename(metadata_filename)),
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
    """
    Dynamic kernel loader for LROC (Lunar Reconnaissance Orbiter Camera) kernels.

    Time bounds are parsed from filenames using a fixed YYYYDDD_YYYYDDD pattern.
    """

    FILENAME_PATTERN = re.compile(r"lrolc_(\d{7})_(\d{7})")

    def parse_kernel_time_bounds(self, filename: str, url: str) -> Dict[str, Time]:
        """
        Extract time interval from filename using regex.

        Parameters:
            filename (str): Filename of the SPICE kernel.
            url (str): URL to the remote file (unused, for interface consistency).

        Returns:
            Dict[str, Time]: Dictionary with `time_start` and `time_stop` keys.

        Raises:
            ValueError: If filename does not match the expected pattern.
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
    Manager of multiple kernel loaders with fallback priority.

    Tries each kernel manager in order until one can handle the requested time.
    This allows composition of multiple independent datasets with partial coverage.
    """

    def __init__(self, kernel_managers: Sequence[LBLDynamicKernelLoader]) -> None:
        """
        Initialize the priority-based kernel manager.

        Parameters:
            kernel_managers (Sequence): Ordered list of kernel loaders to manage.
        """
        self.kernel_managers = kernel_managers

    @property
    def min_loaded_time(self) -> Optional[Time]:
        """
        Get the earliest time covered by any managed loader.

        Returns:
            Optional[Time]: Earliest start time, or None if no loader has data.
        """
        if not self.kernel_managers:
            return None

        return min(km.min_loaded_time for km in self.kernel_managers if km.min_loaded_time is not None)

    @property
    def max_loaded_time(self) -> Optional[Time]:
        """
        Get the latest time covered by any managed loader.

        Returns:
            Optional[Time]: Latest stop time, or None if no loader has data.
        """
        if not self.kernel_managers:
            return None
        return max(km.max_loaded_time for km in self.kernel_managers if km.max_loaded_time is not None)

    def reload_kernels(self, time: Time) -> bool:
        """
        Reload a kernel for the given time from the first capable loader.

        Parameters:
            time (Time): Target time to load.

        Returns:
            bool: True if any loader handled it, False if none could.
        """
        for i, km in enumerate(self.kernel_managers):
            if km.reload_kernels(time):
                logger.debug("Time %s handled by kernel manager #%d", time.iso, i)
                return True
        logger.warning("No kernel manager could handle time %s", time.iso)
        return False

    def unload(self) -> None:
        """
        Unload all kernels from all managed loaders.
        """
        for km in self.kernel_managers:
            km.unload()
