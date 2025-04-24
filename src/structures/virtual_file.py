import requests
import threading
import logging
import time
from typing import Optional, Dict

from io import BytesIO

from src.data_fetchers.config import MAX_RETRIES, DOWNLOAD_TIMEOUT, DOWNLOAD_CHUNK_SIZE
from src.structures import TimeInterval


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)-8s %(message)s")
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
        """
        Download the file in a separate thread, with resume support and retries.
        """
        def _reset_buffer():
            """Drop the partial download and start over."""
            try:
                self.file.close()
            except Exception:
                pass
            self.file = BytesIO()

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
        if self._thread:
            self._done.wait()  # Wait in case we still download
            del self._thread
            self._thread = None
        if self.file:
            self.file.close()
            del self.file
            self.file = BytesIO()
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
