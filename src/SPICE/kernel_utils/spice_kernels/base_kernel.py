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
import random
import requests
import time
import threading
from typing import Optional

import spiceypy as spice
from filelock import FileLock

from src.structures import SharedFileUseLock

from src.SPICE.config import (
    MAX_RETRIES,
    SPICE_CHUNK_SIZE,
    SPICE_TOTAL_TIMEOUT,
    SPICE_KERNEL_LOCK_DOWNLOAD_TIMEOUT,
    KERNEL_LOCK_POLL_INTERVAL,
)

logger = logging.getLogger(__name__)


class BaseKernel:
    """
    Base implementation of a SPICE kernel abstraction.

    Handles downloading, loading, unloading, and file lifecycle management of a single kernel.
    Includes concurrency-safe downloading with file locking and shared usage registration
    to prevent accidental deletion in multi-process setups.

    Intended to be subclassed for kernels with specific behaviors (e.g. LBL parsing, metadata support).
    """

    url: str
    filename: str

    def __init__(self, url: str, filename: str, keep_kernel: bool = True) -> None:
        self._loaded = False
        self.url = url
        self.filename = filename
        self.corrupted = False
        self._keep_kernel = keep_kernel
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        # Obtain the lock for kernel and register the use
        # We do not really care whether the file exists now, but if it does, we ensure other process will not delete it
        self._lock = SharedFileUseLock(self.filename)
        self._lock.register_use()

    @property
    def file_exists(self) -> bool:
        """
        Check whether the kernel file exists locally on disk.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return os.path.exists(self.filename)

    def ensure_downloaded(self) -> None:
        """
        Ensure the kernel file is downloaded locally.

        If not present, triggers a download. Uses locking to prevent concurrent downloads.
        """
        if not self.file_exists:
            self.download_file(self.url, self.filename)

    def unload(self) -> None:
        """
        Unload the kernel from SPICE, release usage lock, and optionally delete the file.

        If the kernel is not loaded, this is a no-op for SPICE but still manages the lock and file cleanup.
        """
        if self._loaded:
            spice.unload(self.filename)
            self._loaded = False
            logger.debug("Unloaded kernel %s", self.filename)
        else:
            logger.debug("Attempted to unload non-loaded kernel %s.", self.filename)
        # Release the lock and delete the file if not needed
        self._lock.release_use()
        if not self._keep_kernel:
            self.delete_file()

    def load(self) -> None:
        """
        Load the kernel into SPICE (furnsh), downloading it if necessary.

        Handles retry and fallback in case of SPICE load errors.
        Ensures the kernel is marked as loaded only on successful SPICE load.
        """
        if not self._loaded:
            self.ensure_downloaded()
            try:
                spice.furnsh(self.filename)
            except:
                spice.reset()
                logger.error("Failed to load kernel %s, trying again", self.filename)
                self.ensure_downloaded()
                spice.furnsh(self.filename)

            self._loaded = True
            logger.debug("[SPICE-LOAD] Loading kernel: %s", self.filename)
        else:
            logger.debug("Kernel %s is already loaded", self.filename)

    def delete_file(self) -> None:
        """
        Schedule the kernel file for asynchronous deletion, respecting shared file use lock.

        Safe to call multiple times; deletion only proceeds when all users have released the file.
        """
        if self.file_exists:
            threading.Thread(target=lambda: self._lock.try_delete_file(), daemon=True).start()
            logger.debug("Scheduled async deletion of kernel file %s", self.filename)
        else:
            logger.debug("Attempted to delete non-existing kernel file %s", self.filename)

    def _verify_file_size(self, path: str, expected: Optional[int]) -> None:
        if expected is None:
            return
        actual = os.path.getsize(path)
        if actual != expected:
            raise ValueError(f"Size mismatch: expected {expected} bytes, got {actual} bytes")

    @staticmethod
    def temp_filename(filename: str):
        """
        Generate a temporary filename for the given final filename.

        Used during partial/resumable downloads.

        Returns:
            str: Path to the temporary file.
        """
        return filename + ".tmp"

    def download_file(self, url: str, filename: str) -> None:
        """
        Safely download a file to disk using locking to prevent concurrent access.

        Handles corrupted downloads, zero-byte files, and ensures the downloaded file is complete
        by checking expected size (if provided by the server).

        Parameters:
            url (str): Source URL of the file.
            filename (str): Target path on local disk.
        """
        tmp_filename = self.temp_filename(filename)
        with FileLock(
            tmp_filename + ".lock", timeout=SPICE_KERNEL_LOCK_DOWNLOAD_TIMEOUT, poll_interval=KERNEL_LOCK_POLL_INTERVAL
        ):
            # Arbitrary retry count for faults on FS level
            for _ in range(3):
                if not os.path.exists(filename):
                    self._download_file(url, filename)

                if os.path.exists(filename):
                    if os.path.getsize(filename) > 0:
                        break
                    else:
                        logger.warning("File %s is empty, retrying download", filename)
                        os.remove(filename)
                        if os.path.exists(tmp_filename):
                            os.remove(tmp_filename)

    def _download_file(self, url: str, filename: str) -> None:
        """
        Internal low-level file downloader with resume support and retry logic.

        Downloads to a temporary file, validates file size against Content-Length or Content-Range,
        and renames to the final file on success. Cleans up on failure.

        Parameters:
            url (str): URL of the remote kernel file.
            filename (str): Local filename (final target).
        """
        retries = 0
        temp_path = self.temp_filename(filename)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyDownloader/1.0)"}
        corruption = False

        while retries < MAX_RETRIES:
            try:
                resume_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                if resume_pos:
                    headers["Range"] = f"bytes={resume_pos}-"

                with requests.get(url, stream=True, timeout=SPICE_TOTAL_TIMEOUT, headers=headers) as r:
                    if r.status_code == 416:  # Requested range not satisfiable
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        resume_pos = 0
                        headers.pop("Range", None)
                        continue  # Retry without resume

                    r.raise_for_status()
                    total_size = None

                    # Determine total expected size
                    if resume_pos:
                        cr = r.headers.get("Content-Range")
                        if cr and "/" in cr:
                            try:
                                total_size = int(cr.split("/")[-1])
                            except ValueError:
                                total_size = None
                        else:
                            logger.warning("Server did not return Content-Range; cannot resume. Restarting download.")
                            resume_pos = 0
                            headers.pop("Range", None)
                            continue  # Retry from beginning
                    else:
                        cl = r.headers.get("Content-Length")
                        total_size = int(cl) if cl and cl.isdigit() else None

                    # Some remote kernels are 0 Bytes?!
                    if total_size is None or total_size <= 0:
                        if not corruption:
                            corruption = True
                        else:
                            logger.warning(f"File {filename} looks corrupted on remote server {url}")
                            self.corrupted = True
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            return

                    mode = "ab" if resume_pos else "wb"
                    with open(temp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=SPICE_CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)

                # Verify the file size
                actual_size = os.path.getsize(temp_path)
                if total_size is not None and actual_size != total_size:
                    raise ValueError(f"Size mismatch: expected {total_size} bytes, got {actual_size} bytes")

                os.rename(temp_path, filename)
                logger.debug("Successfully downloaded %s", url)
                return

            except Exception as e:
                retries += 1
                logger.warning("Attempt %d failed to download %s: %s", retries, url, e)
                sleep_time = max((2**retries), 120) + (random.random() * retries)
                if retries < MAX_RETRIES:
                    time.sleep(sleep_time)
                else:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    logger.error("Failed to download %s after %s attempts", url, MAX_RETRIES)
                    return

    def __repr__(self):
        return f"<BaseKernel filename='{self.filename}'>"
