"""
====================================================
SPICE Kernel Management Module
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

import logging
import re
from typing import Optional

from astropy.time import Time
from tqdm import tqdm


from src.SPICE.config import (
    KERNEL_TIME_KEYS,
)
from src.SPICE.kernel_utils.spice_kernels.base_kernel import BaseKernel
from src.SPICE.kernel_utils.spice_kernels.static_kernels import LBLKernel


logger = logging.getLogger(__name__)


class DynamicKernel(BaseKernel):
    """
    SPICE kernel subclass with a defined valid time interval.

    Provides a method to check whether a specific time lies within the kernel's validity range.
    Intended for dynamic or static kernels whose usage is temporally constrained.
    """

    time_start: Optional[Time]
    time_stop: Optional[Time]

    def __init__(
        self, url: str, filename: str, time_start: Optional[Time], time_stop: Optional[Time], keep_kernel: bool = True
    ):
        """
        Initialize a time-bounded kernel.

        Parameters:
            url (str): Remote URL of the kernel file.
            filename (str): Local path to store the kernel.
            time_start (Optional[Time]): Start of the kernel's valid time range.
            time_stop (Optional[Time]): End of the kernel's valid time range.
            keep_kernel (bool): If False, the kernel file is deleted after unloading.
        """
        super().__init__(url, filename, keep_kernel=keep_kernel)
        self.time_start = time_start
        self.time_stop = time_stop

    def in_interval(self, time: Time) -> bool:
        """
        Check whether a given time lies within the kernel's valid time range.

        Parameters:
            time (Time): Time to evaluate.

        Returns:
            bool: True if the time is within the interval, False otherwise.

        Raises:
            TypeError: If time_start or time_stop is None.
        """
        return self.time_start <= time <= self.time_stop


class LBLDynamicKernel(LBLKernel, DynamicKernel):
    """
    Dynamic SPICE kernel using a corresponding .LBL metadata file to define time validity.

    Combines label parsing (from LBLKernel) with time-bound filtering (from DynamicKernel).
    """

    def __init__(self, url, filename, metadata_url, metadata_filename, keep_kernel: bool = True):
        """
        Initialize an LBL-based dynamic kernel.

        Parameters:
            url (str): Remote URL of the kernel file.
            filename (str): Local kernel file path.
            metadata_url (str): Remote URL of the metadata (.LBL) file.
            metadata_filename (str): Local metadata file path.
            keep_kernel (bool): If False, the kernel file is deleted after unloading.
        """
        LBLKernel.__init__(self, url, filename, metadata_url, metadata_filename, keep_kernel=keep_kernel)
        self.time_start = None
        self.time_stop = None

    def get_time_interval(self, pbar: Optional[tqdm] = None) -> bool:
        """
        Download and parse the associated .LBL metadata to extract time bounds.

        Expected keys (from `KERNEL_TIME_KEYS`) are used to populate `time_start` and `time_stop`.

        Parameters:
            pbar (tqdm): Progress bar instance to update when the interval is successfully parsed.

        Returns:
            bool: True if both time bounds were found and parsed, False otherwise.
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

