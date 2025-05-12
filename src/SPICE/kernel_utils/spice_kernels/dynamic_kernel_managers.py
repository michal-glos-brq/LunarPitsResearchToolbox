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


