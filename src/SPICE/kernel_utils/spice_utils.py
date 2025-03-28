"""
====================================================
SPICE Dynamic Kernel Management
====================================================

Author: Michal Gloš
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

Description:
------------
This module manages dynamic SPICE kernels, ensuring that the correct 
kernels are loaded for the given simulation time. It handles:

1. **Kernel Downloading**: Fetches SPICE kernels from remote repositories.
2. **Kernel Caching**: Maintains an optimal number of loaded kernels.
3. **Time-Based Kernel Switching**: Ensures that the correct CK kernels 
   are active for the given simulation time.
4. **Parallel Downloading**: Uses asyncio to efficiently download metadata 
   and kernels when needed.

Components:
-----------
- **BaseKernel**: Handles static SPICE kernels with basic download and caching.
- **AutoUpdateKernel**: Automatically selects the latest available kernel.
- **LBLDynamicKernel**: Parses `.lbl` metadata files to determine valid time intervals.
- **LBLDynamicKernelLoader**: Loads, manages, and switches dynamic kernels as needed.
- **PriorityKernelManagement**: Prioritizes different sources of kernels.
"""


import os
import logging
import random
import sys
import re
import requests
import time
import asyncio
import aiohttp
import aiofiles
from collections import OrderedDict
from urllib.parse import urljoin
from typing import Optional, List, Tuple, Callable, Iterable, Dict

from astropy.time import Time, TimeDelta
import spiceypy as spice
from tqdm.asyncio import tqdm
from bs4 import BeautifulSoup as bs

sys.path.insert(0, "/".join(__file__.split("/")[:-3]))

from src.global_config import TQDM_NCOLS
from src.SPICE.config import MAX_LOADED_DYNAMIC_KERNELS, MAX_KERNEL_DOWNLOADS

logger = logging.getLogger(__name__)

# Default keys to be used to read .lbl files adjecent to spice kernels
KERNEL_TIME_KEYS = {"filename_key": "^SPICE_KERNEL", "time_start_key": "START_TIME", "time_stop_key": "STOP_TIME"}
SECOND_TIMEDELTA = TimeDelta(1, format="sec")
MAX_RETRIES = 250
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

#####################################################################################################################
#####                                            Single kernel utils                                            #####
#####################################################################################################################

def get_aiohttp_session():
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60, connect=10, sock_connect=10, sock_read=60),
        headers=HEADERS
    )

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

    async def async_download_kernel(self) -> None:
        """Asynchronously download the kernel with a progress bar update"""
        if not self.file_exists:
            await self._async_download_file(self.url, self.filename)

    def unload(self) -> None:
        """Unload the kernel from spiceypy"""
        spice.unload(self.filename)

    # def load(self, load_callbacks: List[Tuple[Callable, Iterable, Dict]] = []) -> None:
    def load(self) -> None:
        """Load the kernel to spiceypy"""
        self.download_kernel()
        # This makes sure kernel is loaded correctly. Could fail once, if it throws error for the second time
        # there is something wrong with the system 
        # try:
        #     for callback, args, kwargs in load_callbacks:
        #         callback(*args, **kwargs)
        # except Exception as e:
        #     for callback, args, kwargs in load_callbacks:
        #         callback(*args, **kwargs)
        spice.furnsh(self.filename)

    def delete_file(self) -> None:
        """Delete the file from disk"""
        os.remove(self.filename)

    def _download_file(self, url, filename) -> None:
        """
        Download the file from the internet with retries and a total timeout of 1 hour.
        If the download fails after all retries, delete the file.
        """
        timeout = 60  # Total timeout in seconds
        retries = 0
        tmp_filename = filename + ".tmp"

        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyDownloader/1.0)"}

        while retries < MAX_RETRIES:
            try:
                with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                    r.raise_for_status()
                    expected_size = r.headers.get("Content-Length")
                    expected_size = int(expected_size) if expected_size and expected_size.isdigit() else None

                    with open(tmp_filename, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192, timeout=timeout):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)

                # Verify file size if available from headers
                actual_size = os.path.getsize(tmp_filename)
                if expected_size is not None and actual_size != expected_size:
                    raise ValueError(f"Size mismatch: expected {expected_size} bytes, got {actual_size} bytes")

                # Rename temp file to final filename after success
                os.rename(tmp_filename, filename)
                logger.debug("Successfully downloaded %s", url)
                return

            except Exception as e:
                retries += 1
                logger.warning("Attempt %s failed to download %s: %s", retries, url, e)
                # Add exponential backoff with jitter to avoid hammering the server
                sleep_time = max((2 ** retries) + (random.random() * retries), 120)
                if retries < MAX_RETRIES:
                    time.sleep(sleep_time)
                else:
                    if os.path.exists(tmp_filename):
                        os.remove(tmp_filename)
                    logger.error("Failed to download %s after %s attempts", url, MAX_RETRIES)
                    return


    def _parse_total_size(self, response):
        """
        If the server supports resume, it should return a Content-Range header of the form:
           "bytes start-end/total"
        """
        cr = response.headers.get("Content-Range")
        if cr:
            try:
                return int(cr.split("/")[-1])
            except Exception:
                return None
        # Fallback: use Content-Length (for full downloads)
        cl = response.headers.get("Content-Length")
        return int(cl) if cl and cl.isdigit() else None

    async def _async_download_file(self, url, filename) -> None:
        """
        Asynchronously download the file with retries and per-chunk timeouts.
        If a temporary file exists, resume the download from where it left off.
        Each attempt uses a new ClientSession that is closed automatically.
        If the download fails after max retries, the temporary file is deleted.
        """
        retries = 0
        tmp_filename = filename + ".tmp"

        while retries < MAX_RETRIES:
            try:
                # Check for resume position.
                resume_pos = os.path.getsize(tmp_filename) if os.path.exists(tmp_filename) else 0
                headers = {}
                if resume_pos:
                    headers["Range"] = f"bytes={resume_pos}-"

                # Create a new session that will close automatically.
                async with get_aiohttp_session() as session:
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 416:
                            if os.path.exists(tmp_filename):
                                os.remove(tmp_filename)
                        response.raise_for_status()

                        # Determine the total expected size.
                        if resume_pos:
                            total_size = self._parse_total_size(response)
                            # If no Content-Range is returned, assume resume not supported.
                            if total_size is None:
                                logger.warning("Server did not return Content-Range; cannot resume. Restarting download.")
                                resume_pos = 0
                                total_size = int(response.headers.get("Content-Length")) if response.headers.get("Content-Length", "").isdigit() else None
                        else:
                            total_size = int(response.headers.get("Content-Length")) if response.headers.get("Content-Length", "").isdigit() else None

                        mode = "ab" if resume_pos else "wb"
                        async with aiofiles.open(tmp_filename, mode) as f:
                            while True:
                                try:
                                    # Read up to 8192 bytes with a per-read timeout (15 seconds)
                                    chunk = await asyncio.wait_for(response.content.read(8192), timeout=15)
                                except asyncio.TimeoutError:
                                    raise Exception("Stalled during chunk read")
                                if not chunk:
                                    break
                                await f.write(chunk)

                # Verify file size if known.
                actual_size = os.path.getsize(tmp_filename)
                if total_size is not None and actual_size != total_size:
                    raise ValueError(f"Size mismatch: expected {total_size} bytes, got {actual_size} bytes")

                os.rename(tmp_filename, filename)
                logger.debug("Successfully downloaded %s", url)
                return

            except Exception as e:
                retries += 1
                # This is expected, for now will be left on debug level.
                logger.debug("Attempt %s failed to download %s: %s", retries, url, e)
                # Capped exponential backoff (max 120 seconds)
                sleep_time = min((2 ** retries) + (random.random() * retries), 120)
                if retries < MAX_RETRIES:
                    await asyncio.sleep(sleep_time)
                else:
                    if os.path.exists(tmp_filename):
                        os.remove(tmp_filename)
                    logger.error("Failed to download %s after %s attempts", url, MAX_RETRIES)
                    return


class AutoUpdateKernel(BaseKernel):

    url: str
    filename: str

    def __init__(self, folder_url: str, folder_path: str, regex: str):
        self.regex = re.compile(regex)
        response = requests.get(folder_url)
        response.raise_for_status()
        soup = bs(response.text, "html.parser")
        links = soup.find_all("a")
        filename = sorted([link.get("href") for link in links if self.regex.search(link.get("href") or "")], reverse=True)[0]
        url = urljoin(folder_url, filename)
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

    async def async_download_metadata(self) -> None:
        """Asynchronously download the metadata with a progress bar update"""
        if not self.metadata_exists:
            await self._async_download_file(self.metadata_url, self.metadata_filename)

    def delete_metadata(self) -> None:
        """Delete the metadata file from disk"""
        os.remove(self.metadata_filename)


class StaticKernelManager:

    def __init__(self, kernel_objects: OrderedDict[str, List[BaseKernel]]):
        # We want to aggregate all kernels in one list while retaining its order
        self.kernel_pool = [kernel for kernels in kernel_objects.values() for kernel in kernels]

        semaphore = asyncio.Semaphore(MAX_KERNEL_DOWNLOADS)

        async def download_kernel(kernel: BaseKernel, pbar: tqdm) -> None:
            async with semaphore:
                await kernel.async_download_kernel()
                pbar.update(1)

        pbar = tqdm(total=len(self.kernel_pool), desc="Downloading static kernels", ncols=TQDM_NCOLS)

        tasks = [download_kernel(kernel, pbar) for kernel in self.kernel_pool]
        asyncio.run(asyncio.gather(*tasks))
        pbar.close()

    def load(self):
        for kernel in self.kernel_pool:
            kernel.load()

    def unload(self):
        for kernel in self.kernel_pool:
            kernel.unload()


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

    async def get_time_interval(self, semaphore, pbar: tqdm) -> None:
        """
        Get the time interval from the metadata file

        Is async so it could be gathered with asyncio

        semaphore: asyncio.Semaphore to limit the number of concurrent downloads
        pbar: tqdm.asyncio.tqdm, goes up 1 per file
        """
        async with semaphore:
            await self.async_download_metadata()

        pattern = re.compile(r"(\S+)\s*=\s*(.+)")
        with open(self.metadata_filename, "r") as f:
            content = f.read()
        metadata = {m.group(1): m.group(2).strip('"') for m in pattern.finditer(content)}
        if not all(key in metadata for key in KERNEL_TIME_KEYS.values()):
            logger.warning("Skipping %s, insufficient data", self.metadata_filename)
            return
        self.time_start = Time(metadata[KERNEL_TIME_KEYS["time_start_key"]], format="isot", scale="utc")
        self.time_stop = Time(metadata[KERNEL_TIME_KEYS["time_stop_key"]], format="isot", scale="utc")
        if pbar:
            pbar.update(1)


class DynamicKernel(BaseKernel):

    def __init__(self, url: str, filename: str, time_start: Time, time_stop: Time) -> None:
        super().__init__(url, filename)
        self.time_start = time_start
        self.time_stop = time_stop

    def in_interval(self, time: Time) -> bool:
        """
        Check if the given time is within the time interval of the kernel
        """
        return self.time_start <= time <= self.time_stop


class DynamicKernelManager:

    @property
    def min_loaded_time(self) -> Time:
        return self.kernel_pool[0].time_start

    @property
    def max_loaded_time(self) -> Time:
        return self.kernel_pool[-1].time_stop

    def get_kernel_pool_kwargs(self, filename, url) -> Dict:
        """
        Get the kernel pool kwargs for the given filename and url
        """
        ...

    def __init__(
        self,
        path: str,
        base_url: str,
        regex: str,
        files_persist: bool = True,
        pre_download_kernels: bool = True,
        # load_callbacks: List[Tuple[Callable, Iterable, Dict]] = [],
    ):
        """
        path: local path to dynamic kernel folder
        base_url: remote location of dynamic kernels
        regex: filter all files from local path to get desired kernels
        files_persist: if True, downloaded files are stored on disk. If false, once data are unloaded,
                    corresponding files are deleted
        pre_download_kernels: if True, all kernels are downloaded at initialization (if not already on the disk)
        load_callbakcs: list of tuples with callback functions, their arguments and keyword arguments. Checks for correct
            loading of kernels. Could throw exception once to load it properly, throwing 2 means something went wrong.
            First element of the iterable is reserved for et (placeholder is needed when defining the callbacks)
            (commented out for now, but in case it would be needed, uncommenting the code will bring it back to life)
        """
        self.regex = re.compile(regex)
        self.base_url = base_url
        self.files_persist = files_persist
        self.path = path
        # self.load_callbacks = load_callbacks
        os.makedirs(self.path, exist_ok=True)

        self.active_kernel_id = -1
        self.loaded_kernels = []
        self.load_metadata()

        if pre_download_kernels:
            self.download_kernels()


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
                **self.get_kernel_pool_kwargs(kernel_filename, kernel_url)
            )
            for kernel_url, kernel_filename in zip(
                kernel_urls, kernel_filenames
            )
        ]
        self.kernel_pool_len = len(self.kernel_pool)
        self.kernel_pool = sorted(self.kernel_pool, key=lambda x: x.time_start)

    def download_kernels(self) -> None:
        """
        Downloads all kernels that are not already on disk
        """
        semaphore = asyncio.Semaphore(MAX_KERNEL_DOWNLOADS)

        async def download_kernel(kernel: LBLDynamicKernel, pbar: tqdm) -> None:
            async with semaphore:
                await kernel.async_download_kernel()
                pbar.update(1)

        pbar = tqdm(total=len(self.kernel_pool), desc="Downloading dynamic kernels", ncols=TQDM_NCOLS)

        tasks = [download_kernel(kernel, pbar) for kernel in self.kernel_pool]
        asyncio.run(asyncio.gather(*tasks))
        pbar.close()

    def load_new_kernel(self, _id: int, time: Time) -> None:
        """
        Load a new kernel to spiceypy with all needed actions

        _id points to kernel list
        """
        kernel = self.kernel_pool[_id]
        self.active_kernel_id = _id

        # for callback in self.load_callbacks:
        #     callback[1][0] = spice.utc2et(time.utc.iso)
        # kernel.load(load_callbacks=self.load_callbacks)
        kernel.load()

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
            self.load_new_kernel(self.active_kernel_id + 1, time)
            return True
        else:
            for i, kernel in enumerate(self.kernel_pool):
                if kernel.in_interval(time):
                    self.load_new_kernel(i, time)
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

class LROCDynamicKernelLoader(DynamicKernelManager):

    def get_kernel_pool_kwargs(self, filename, url) -> Dict:
        match = re.search(r"lrolc_(\d{7})_(\d{7})", filename)
        if not match:
            raise ValueError(f"Filename does not match expected pattern: {filename}")
    
        start_doy, end_doy = match.groups()
        start_doy, end_doy = start_doy[:4] + ":" + start_doy[4:], end_doy[:4] + ":" + end_doy[4:]
        t_start = Time(start_doy, format="yday", scale="utc")
        t_end = Time(end_doy, format="yday", scale="utc")
        return {"time_start": t_start, "time_stop": t_end}

class LBLDynamicKernelLoader(DynamicKernelManager):

    def __init__(
        self,
        path: str,
        base_url: str,
        regex: str,
        metadata_regex: str,
        files_persist: bool = True,
        pre_download_kernels: bool = True,
        # load_callbacks: List[Tuple[Callable, Iterable, Dict]] = [],
    ):
        """
        path: local path to dynamic kernel folder
        base_url: remote location of dynamic kernels
        regex: filter all files from local path to get desired kernels
        metadata_regex: filter all files with metadata, implicitly .lbl files
        files_persist: if True, downloaded files are stored on disk. If false, once data are unloaded,
                    corresponding files are deleted
        pre_download_kernels: if True, all kernels are downloaded at initialization (if not already on the disk)
        load_callbakcs: list of tuples with callback functions, their arguments and keyword arguments. Checks for correct
            loading of kernels. Could throw exception once to load it properly, throwing 2 means something went wrong.
            First element of the iterable is reserved for et (placeholder is needed when defining the callbacks)
            (commented out for now, but in case it would be needed, uncommenting the code will bring it back to life)
        """
        self.metadata_regex = re.compile(metadata_regex)
        super().__init__(path, base_url, regex, files_persist, pre_download_kernels)

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
            )
            for kernel_url, kernel_filename, metadata_url, metadata_filename in zip(
                kernel_urls, kernel_filenames, metadata_urls, metadata_filenames
            )
        ]
        self.kernel_pool_len = len(self.kernel_pool)
        semaphore = asyncio.Semaphore(MAX_KERNEL_DOWNLOADS)

        pbar = tqdm(total=len(self.kernel_pool), desc="Downloading dynamic kernel metadata", ncols=TQDM_NCOLS)

        tasks = [kernel.get_time_interval(semaphore, pbar) for kernel in self.kernel_pool]
        asyncio.run(asyncio.gather(*tasks))
        
        pbar.close()

        self.kernel_pool = sorted(self.kernel_pool, key=lambda x: x.time_start)


class PriorityKernelManagement:

    def __init__(self, kernel_managers: List[LBLDynamicKernelLoader]) -> None:
        """kernel_managers: list of kernel managers, ordered by priority"""
        self.kernel_managers = kernel_managers

    ### Priority means we want at leasst one kernel to be present at the given time.
    ### So we looking at min and max, not for minimal maximal and maximal minimal time :)
    @property
    def min_loaded_time(self) -> Time:
        return min(km.min_loaded_time for km in self.kernel_managers)

    @property
    def max_loaded_time(self) -> Time:
        return max(km.max_loaded_time for km in self.kernel_managers)

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
