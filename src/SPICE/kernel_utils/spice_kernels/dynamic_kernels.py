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
from src.SPICE.kernel_utils.spice_kernels.base_static_kernel import BaseKernel
from src.SPICE.kernel_utils.spice_kernels.static_kernels import LBLKernel


logger = logging.getLogger(__name__)


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
