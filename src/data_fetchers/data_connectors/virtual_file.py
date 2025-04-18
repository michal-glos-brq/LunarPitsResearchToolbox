import requests
import threading
import logging
import random
import time
from typing import Optional, Dict

from io import BytesIO

from src.data_fetchers.config import MAX_RETRIES, DOWNLOAD_TIMEOUT, DOWNLOAD_CHUNK_SIZE
from src.data_fetchers.interval_manager import TimeInterval


logger = logging.getLogger(__name__)


class VirtualFile:
    """
    Represents a virtual file with a name and content.
    Used for testing purposes to simulate file downloads.
    """

    def __init__(self, url: str, time_interval: TimeInterval, metadata: Optional[Dict] = None):
        self.metadata = metadata
        self.url = url
        self.interval = time_interval
        self.file = BytesIO()
        self.data = None
        self._thread = None
        # Thread-safe flag
        self._done = threading.Event()

    def _download_worker(self):
        """Download the file in a separate thread. Used for alrger files"""
        retries = 0
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyDownloader/1.0)"}
        corruption = False

        logger.info(f"Starting download for {self.url}")

        while retries < MAX_RETRIES:
            try:
                resume_pos = self.file.tell()  # Position in BytesIO buffer
                if resume_pos:
                    headers["Range"] = f"bytes={resume_pos}-"

                with requests.get(self.url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as r:
                    if r.status_code == 416:  # Range not satisfiable
                        logger.warning(f"Range not satisfiable, resetting download for {self.url}")
                        self.file = BytesIO()
                        resume_pos = 0
                        headers.pop("Range", None)
                        continue

                    r.raise_for_status()

                    # Determine expected total size
                    if resume_pos:
                        cr = r.headers.get("Content-Range")
                        if cr and "/" in cr:
                            try:
                                self._expected_size = int(cr.split("/")[-1])
                            except ValueError:
                                self._expected_size = None
                        else:
                            logger.warning(f"Server did not return Content-Range, restarting download for {self.url}")
                            self.file = BytesIO()
                            resume_pos = 0
                            headers.pop("Range", None)
                            continue
                    else:
                        cl = r.headers.get("Content-Length")
                        self._expected_size = int(cl) if cl and cl.isdigit() else None

                    if self._expected_size is None or self._expected_size <= 0:
                        if not corruption:
                            corruption = True
                            logger.warning(f"Remote file seems empty or corrupt, will retry: {self.url}")
                        else:
                            logger.error(f"File {self.url} looks corrupted on remote server, aborting.")
                            self.corrupted = True
                            return

                    # Read and write to in-memory buffer
                    for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            self.file.write(chunk)

                # Verify size
                actual_size = self.file.tell()
                if self._expected_size is not None and actual_size != self._expected_size:
                    raise ValueError(f"Size mismatch: expected {self._expected_size} bytes, got {actual_size} bytes")

                logger.debug(f"Successfully downloaded {self.url} into memory")
                self.file.seek(0)
                return

            except Exception as e:
                retries += 1
                logger.warning(f"Attempt {retries} failed to download {self.url}: {e}")
                sleep_time = max((2**retries), 120) + (random.random() * retries)
                if retries < MAX_RETRIES:
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to download {self.url} after {MAX_RETRIES} attempts")
                    self.corrupted = True
                    return
            finally:
                self._done.set()

    def download(self):
        """Start download in background thread."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._download_worker, daemon=True)
            self._thread.start()

    def unload(self):
        """Unload the file content."""
        logger.info(f"Unloading {self.url}")
        if self.file:
            self.file.close()
            del self.file
            self.file = None
        if self._thread:
            del self._thread
            self._thread = None
        if self.data:
            del self.data
            self.data = None
        self._done.clear()

    def is_done(self):
        """Check if download is completed."""
        return self._done.is_set()

    def wait_to_be_downloaded(self, timeout=None):
        """Block until download is finished."""
        self._done.wait(timeout=timeout)
