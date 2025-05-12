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
from typing import List
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.global_config import TQDM_NCOLS, SUPRESS_TQDM
from src.SPICE.config import MAX_KERNEL_DOWNLOADS
from src.SPICE.kernel_utils.spice_kernels.base_static_kernel import BaseKernel

logger = logging.getLogger(__name__)


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