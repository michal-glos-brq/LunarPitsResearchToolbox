import os
import requests
import threading
import logging
import time
from datetime import datetime
from typing import Optional, Dict

from io import BytesIO

from .interval_manager import TimeInterval
from ..global_config import SAVE_DATAFILES_TO_RAM, HDD_BASE_PATH

MAX_RETRIES = 128
DOWNLOAD_TIMEOUT = 60 * 60 * 2 # 2 hours
DOWNLOAD_CHUNK_SIZE = 8 * 1024 # 8 KB


logger = logging.getLogger(__name__)

DATAFILES_TMP_DESTINATION = os.path.join(HDD_BASE_PATH, "data_tmp")
os.makedirs(DATAFILES_TMP_DESTINATION, exist_ok=True)


class VirtualFile:
    """
    Represents a virtual file with a name and content.
    Used for testing purposes to simulate file downloads.
    """

    def __init__(self, url: str, time_interval: TimeInterval, metadata: Optional[Dict] = None):
        self.metadata = metadata
        self.url = url
        self.interval = time_interval
        self._thread = None
        self._done = threading.Event()
        self.corrupted = False
        self.data = None

        # initialize buffer or file handle
        self._make_file_handle()


    def _make_file_handle(self):
        """Create either a BytesIO or a real file on disk with a hashed name."""
        if SAVE_DATAFILES_TO_RAM:
            self.file_path = None
            self.file = BytesIO()
        else:
            name = f"{datetime.now().isoformat()}_{os.getpid()}".replace(":", "-").replace(".", "-")
            path = os.path.join(DATAFILES_TMP_DESTINATION, name)
            self.file_path = path
            # open for write+binary, truncate if exists
            self.file = open(path, "w+b")


    def _download_worker(self):
        """
        Download the file in a separate thread, with resume support and retries.
        """
        def _reset_buffer():
            try:
                # close current handle
                self.file.close()
                # if on-disk, remove the partial file
                if not SAVE_DATAFILES_TO_RAM and self.file_path and os.path.exists(self.file_path):
                    os.remove(self.file_path)
            except Exception:
                pass
            # create a brand-new handle (in RAM or on disk)
            self._make_file_handle()

        retries = 0
        self.corrupted = False
        logger.info(f"Starting download for {self.url}")

        try:
            while retries < MAX_RETRIES:
                resume_pos = self.file.tell()
                headers = {"User-Agent": "Mozilla/5.0 (compatible; MyDownloader/1.0)"}
                if resume_pos:
                    headers["Range"] = f"bytes={resume_pos}-"

                # 1) Attempt the HTTP GET
                try:
                    resp = requests.get(self.url, stream=True,
                                        timeout=DOWNLOAD_TIMEOUT,
                                        headers=headers)
                except Exception as e:
                    logger.warning(f"[{retries+1}/{MAX_RETRIES}] Network error: {e}")
                    retries += 1
                    time.sleep(min(2**retries, 60))
                    continue

                # 2) Handle 416 → restart from zero
                if resp.status_code == 416:
                    logger.warning(f"[{retries+1}/{MAX_RETRIES}] Range not satisfiable; resetting buffer")
                    _reset_buffer()
                    retries += 1
                    time.sleep(min(2**retries, 60))
                    continue

                # 3) Any other HTTP error?
                if not resp.ok:
                    logger.warning(f"[{retries+1}/{MAX_RETRIES}] HTTP {resp.status_code}")
                    retries += 1
                    time.sleep(min(2**retries, 60))
                    continue

                # 4) Figure out expected size
                if resume_pos and resp.status_code == 206:
                    cr = resp.headers.get("Content-Range", "")
                    if "/" in cr:
                        try:
                            expected = int(cr.split("/",1)[1])
                        except ValueError:
                            expected = None
                    else:
                        logger.warning("Invalid Content-Range on resume; restarting")
                        _reset_buffer()
                        retries += 1
                        time.sleep(min(2**retries, 60))
                        continue
                else:
                    cl = resp.headers.get("Content-Length", "")
                    expected = int(cl) if cl.isdigit() else None

                # 5) Stream into buffer
                for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        self.file.write(chunk)

                actual = self.file.tell()
                if actual == 0:
                    logger.warning(f"[{retries+1}/{MAX_RETRIES}] No data received; restarting")
                    _reset_buffer()
                    retries += 1
                    time.sleep(min(2**retries, 60))
                    continue

                # 6) Verify size if known
                if expected is not None and actual != expected:
                    logger.warning(
                        f"[{retries+1}/{MAX_RETRIES}] Size mismatch: "
                        f"expected {expected}, got {actual}; restarting"
                    )
                    _reset_buffer()
                    retries += 1
                    time.sleep(min(2**retries, 60))
                    continue

                # 7) Success!
                self.file.seek(0)
                logger.info(f"Downloaded {self.url} ({actual} bytes)")
                return

            # Retries exhausted
            logger.error(f"Failed to download after {MAX_RETRIES} attempts: {self.url}")
            self.corrupted = True

        finally:
            # Always signal done, exactly once
            self._done.set()


    def assert_download_ok(self):
        if self.corrupted:
            raise RuntimeError(f"Download failed for {self.url}")


    def download(self):
        """Start download in background thread."""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._download_worker, daemon=True)
            self._thread.start()

    def unload(self):
        """Unload the file content."""
        logger.info(f"Unloading {self.url}")

        # Clean the thread
        if self._thread:
            self._done.wait()  # Wait in case we still download
            del self._thread
            self._thread = None
        # Clean the file
        if self.file:
            self.file.close()
            del self.file
            if not SAVE_DATAFILES_TO_RAM:
                if os.path.exists(self.file_path):
                    os.remove(self.file_path)
            self._make_file_handle()

        # Clean the buffer
        if self.data is not None:
            del self.data
            self.data = None
        self._done.clear()

    def is_done(self):
        """Check if download is completed."""
        return self._done.is_set()

    def wait_to_be_downloaded(self, timeout=None):
        """Block until download is finished."""
        self._done.wait(timeout=timeout)
        self.assert_download_ok()

    def __repr__(self):
        if self._thread is None:
            status = "not started"
        elif self.is_done():
            status = "done"
        else:
            status = "downloading"

        corrupted = " (CORRUPTED)" if getattr(self, "corrupted", False) else ""
        size = self.file.tell() if hasattr(self, "file") and self.file else 0
        return (
            f"<VirtualFile url='{self.url}' "
            f"interval={self.interval} "
            f"status={status}{corrupted} "
            f"size={size}B>"
        )
