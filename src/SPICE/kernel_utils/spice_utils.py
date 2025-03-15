"""
Define a class representing (usually CK) dynamic in time SPICE kernels amd 
making sure the correct kernels are loaded for the correct time.

Is also responsible for downloading and managing the kernels.
"""

import os
import logging
import sys
import re
import requests
import asyncio
from typing import Optional, List

from astropy.time import Time
import spiceypy as spice
from tqdm.asyncio import tqdm
from bs4 import BeautifulSoup as bs

sys.path.insert(0, "/".join(__file__.split("/")[:-3]))

from src.global_config import TQDM_NCOLS, MAX_CONCURRENT_DOWNLOADS
from src.SPICE.config import MAX_LOADED_DYNAMIC_KERNELS

logger = logging.getLogger(__name__)

# Default keys to be used to read .lbl files adjecent to spice kernels
KERNEL_TIME_KEYS = {"filename_key": "^SPICE_KERNEL", "time_start_key": "START_TIME", "time_stop_key": "STOP_TIME"}


#####################################################################################################################
#####                                            Single kernel utils                                            #####
#####################################################################################################################


class BaseKernel:
    """Base implementation of SPICE kernel"""

    url: str
    filename: str

    def __init__(self, url: str, filename: str) -> None:
        self.url = url
        self.filename = filename
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

    @property
    def file_exists(self) -> bool:
        """Check if the file exists on disk"""
        return os.path.exists(self.filename)

    def download_kernel(self) -> None:
        """Download the kernel from the internet"""
        if not self.file_exists:
            self._download_file(self.url, self.filename)

    def unload(self) -> None:
        """Unload the kernel from spiceypy"""
        spice.unload(self.filename)

    def delete_file(self) -> None:
        """Delete the file from disk"""
        os.remove(self.filename)

    def _download_file(self, url, filename) -> None:
        """
        Download the file from the interne
        """
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                file_iterator = r.iter_content(chunk_size=8192)
                for chunk in file_iterator:
                    f.write(chunk)


class AutoUpdateKernel(BaseKernel):

    url: str
    filename: str

    def __init__(self, folder_url: str, folder_path: str, regex: str):
        self.regex = re.compile(regex)
        response = requests.get(folder_url)
        response.raise_for_status()
        soup = bs(response.text, "html.parser")
        links = soup.find_all("a")
        filename = sorted([link.get("href") for link in links if self.regex.search(link.get("href") or "")], desc=True)[0]
        url = f"{folder_url}{filename}"
        super().__init__(url, os.path.join(folder_path, filename))
        


class LBLKernel(BaseKernel):
    
    def __init__(self, url: str, filename: str, metadata_url: str, metadata_filename: str) -> None:
        super().__init__(url, filename)
        self.metadata_url = metadata_url
        self.metadata_filename = metadata_filename

    @property
    def metadata_exists(self) -> bool:
        """Check if the metadata file exists on disk"""
        return os.path.exists(self.metadata_filename)

    def download_metadata(self) -> None:
        """Download the metadata file from the internet"""
        if not self.metadata_exists:
            self._download_file(self.metadata_url, self.metadata_filename)

    def delete_metadata(self) -> None:
        """Delete the metadata file from disk"""
        os.remove(self.metadata_filename)


class StaticKernelManager:

    def __init__(self, kernel_objects: BaseKernel):
        self.kernel_objects = kernel_objects

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        async def download_kernel(kernel: LBLDynamicKernel) -> None:
            with semaphore:
                kernel.download_kernel()

        tasks = [download_kernel(kernel) for _kernels in self.kernel_objects.values() for kernel in _kernels]
        tqdm.gather(*tasks, desc="Downloading lone kernels", ncols=TQDM_NCOLS)


    def furnsh(self):
        for kernels in self.kernel_objects.values():
            for kernel in kernels:
                spice.furnsh(kernel.filename)


#####################################################################################################################
#####                                            Dynamic kernel utils                                           #####
#####################################################################################################################


class LBLDynamicKernel(LBLKernel):
    """
    Class representing a single SPICE kernel file, which also has .lbl metadata files
    """

    metadata_url: str
    metadata_filename: str
    time_start: Optional[Time]
    time_stop: Optional[Time]

    def __init__(self, url: str, filename: str, metadata_url: str, metadata_filename: str) -> None:
        super().__init__(url, filename, metadata_url, metadata_filename)
        self.time_start = None
        self.time_stop = None

    def in_interval(self, time: Time) -> bool:
        """
        Check if the given time is within the time interval of the kernel
        """
        return self.time_start <= time <= self.time_stop



    async def get_time_interval(self, semaphore) -> None:
        """
        Get the time interval from the metadata file

        Is async so it could be gathered with asyncio

        semaphore: asyncio.Semaphore to limit the number of concurrent downloads
        """
        with semaphore:
            self.download_metadata()

        pattern = re.compile(r"(\S+)\s*=\s*(.+)")
        with open(self.metadata_filename, "r") as f:
            content = f.read()
        metadata = {m.group(1): m.group(2).strip('"') for m in pattern.finditer(content)}
        if not all(key in metadata for key in KERNEL_TIME_KEYS.values()):
            logger.warning("Skipping %s, insufficient data", self.metadata_filename)
            return
        self.time_start = Time(metadata[KERNEL_TIME_KEYS["time_start_key"]], format="isot", scale="utc")
        self.time_stop = Time(metadata[KERNEL_TIME_KEYS["time_stop_key"]], format="isot", scale="utc")


class LBLDynamicKernelLoader:
    @property
    def min_loaded_time(self) -> Time:
        return self.kernel_pool[0].time_start

    @property
    def max_loaded_time(self) -> Time:
        return self.kernel_pool[-1].time_stop

    def __init__(
        self,
        path: str,
        base_url: str,
        regex: str,
        metadata_regex: str,
        files_persist: bool = True,
        pre_download_kernels: bool = True,
    ):
        """
        path: local path to dynamic kernel folder
        base_url: remote location of dynamic kernels
        regex: filter all files from local path to get desired kernels
        metadata_regex: filter all files with metadata, implicitly .lbl files
        files_persist: if True, downloaded files are stored on disk. If false, once data are unloaded,
                    corresponding files are deleted
        pre_download_kernels: if True, all kernels are downloaded at initialization (if not already on the disk)
        """
        self.regex = re.compile(regex)
        self.metadata_regex = re.compile(metadata_regex)
        self.base_url = base_url
        self.files_persist = files_persist
        self.path = path
        os.makedirs(self.path, exist_ok=True)

        self.active_kernel_id = -1
        self.loaded_kernels = []
        self.load_metadata(concurrency=MAX_CONCURRENT_DOWNLOADS)

        if pre_download_kernels:
            self.download_kernels(concurrency=MAX_CONCURRENT_DOWNLOADS)

    def download_kernels(self, concurrency: int) -> None:
        """
        Downloads all kernels that are not already on disk
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def download_kernel(kernel: LBLDynamicKernel) -> None:
            with semaphore:
                kernel.download_kernel()

        tasks = [download_kernel(kernel) for kernel in self.kernel_pool]
        tqdm.gather(*tasks, desc="Downloading dynamic kernels", ncols=TQDM_NCOLS)

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
        kernel_urls = [f"{self.base_url}{filename}" for filename in kernel_filenames]
        metadata_urls = [f"{self.base_url}{filename}" for filename in metadata_filenames]

        self.kernel_pool: List[LBLDynamicKernel] = [
            LBLDynamicKernel(
                kernel_url,
                os.path.join(self.path, kernel_filename),
                metadata_url,
                os.path.join(self.path, metadata_filename),
            )
            for kernel_url, kernel_filename, metadata_url, metadata_filename in zip(
                kernel_urls, kernel_filenames, metadata_urls, metadata_filenames
            )
        ]
        self.kernel_pool_len = len(self.kernel_pool)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        tasks = [kernel.get_time_interval(semaphore) for kernel in self.kernel_pool]
        tqdm.gather(*tasks, desc="Downloading dynamic kernel metadata", ncols=TQDM_NCOLS)
        self.kernel_pool = sorted(self.kernel_pool, key=lambda x: x.time_start)

    def load_new_kernel(self, _id: int) -> None:
        """
        Load a new kernel to spiceypy with all needed actions

        _id points to kernel list
        """
        kernel = self.kernel_pool[_id]
        self.active_kernel_id = _id
        kernel.download_kernel()
        spice.furnsh(kernel.filename)
        self.loaded_kernels.append(kernel)
        if len(self.loaded_kernels) > MAX_LOADED_DYNAMIC_KERNELS:
            spice.unload(self.loaded_kernels[0].filename)
            if not self.files_persist:
                self.loaded_kernels[0].delete_file()
            self.loaded_kernels.pop(0)


    def reload_kernels(self, time: Time) -> bool:
        """
        Reloads kernels for given time
        """
        if self.loaded_kernels and self.loaded_kernels[-1].in_interval(time):
            return True
        elif self.active_kernel_id < self.kernel_pool_len - 1 and self.kernel_pool[self.active_kernel_id + 1].in_interval(time):
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
            spice.unload(kernel.filename)
            if not self.files_persist:
                kernel.delete_file()
        self.loaded_kernels = []
        self.active_kernel_id = -1
        self.kernel_pool = []

class PriorityKernelManagement:
    def __init__(self, kernel_managers: List[LBLDynamicKernelLoader]) -> None:
        """kernel_managers: list of kernel managers, ordered by priority"""
        self.kernel_managers = kernel_managers

    def reload_kernels(self, time: Time) -> bool:
        """
        Reloads kernels for given time
        """
        for kernel_manager in self.kernel_managers:
            if kernel_manager.reload_kernels(time):
                return True
        return False

    def unload(self) -> None:
        """
        Unload all loaded kernels
        """
        for kernel_manager in self.kernel_managers:
            kernel_manager.unload()
