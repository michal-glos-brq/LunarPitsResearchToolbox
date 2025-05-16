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

from bs4 import BeautifulSoup as bs

from src.SPICE.kernel_utils.spice_kernels.base_kernel import BaseKernel

logger = logging.getLogger(__name__)



class AutoUpdateKernel(BaseKernel):
    """
    Kernel that automatically selects and downloads the newest file from a remote folder.

    Requires filenames to contain sortable time/date components.
    The newest matching file (determined by lexicographic sort) is selected based on regex match.
    """

    url: str
    filename: str

    def __init__(self, folder_url: str, folder_path: str, regex: str, keep_kernel: bool = True):
        """
        Initialize the auto-update kernel.

        Parameters:
            folder_url (str): URL of the remote directory to search.
            folder_path (str): Local path where the selected kernel will be stored.
            regex (str): Regex pattern to filter relevant kernel filenames.
            keep_kernel (bool): If False, deletes the file after unloading.
        
        Raises:
            ValueError: If no matching files are found.
        """
        self.regex = re.compile(regex)
        response = requests.get(folder_url)
        response.raise_for_status()
        soup = bs(response.text, "html.parser")
        links = soup.find_all("a")
        matches = [link.get("href") for link in links if self.regex.search(link.get("href") or "")]
        if not matches:
            raise ValueError(f"No matching kernel found in {folder_url} with regex {regex}")
        filename = sorted(matches, reverse=True)[0]
        url = urljoin(folder_url, filename)
        super().__init__(url, os.path.join(folder_path, filename), keep_kernel=keep_kernel)


class LBLKernel(BaseKernel):
    """
    SPICE kernel associated with a .LBL metadata file.

    Manages the download and deletion of the corresponding metadata file.
    """


    def __init__(
        self, url: str, filename: str, metadata_url: str, metadata_filename: str, keep_kernel: bool = True
    ) -> None:
        """
        Initialize a kernel with its metadata pair.

        Parameters:
            url (str): Remote URL of the kernel.
            filename (str): Local path to store the kernel.
            metadata_url (str): Remote URL of the metadata file.
            metadata_filename (str): Local path to store the metadata.
            keep_kernel (bool): If False, deletes the kernel file after unloading.
        """
        BaseKernel.__init__(self, url, filename, keep_kernel=keep_kernel)
        self.metadata_url = metadata_url
        self.metadata_filename = metadata_filename

    @property
    def metadata_exists(self) -> bool:
        """
        Check whether the metadata file exists locally.

        Returns:
            bool: True if metadata is present on disk, False otherwise.
        """
        return os.path.exists(self.metadata_filename)

    def download_metadata(self) -> None:
        """
        Download the metadata (.LBL) file if not already present.

        Uses the same locking and download logic as BaseKernel.
        """
        if not self.metadata_exists:
            logger.debug("Downloading metadata: %s", self.metadata_filename)
            self.download_file(self.metadata_url, self.metadata_filename)
        else:
            logger.debug("Metadata already exists: %s", self.metadata_filename)

    def delete_metadata(self) -> None:
        """
        Delete the local metadata file from disk if it exists.
        Logs the result.
        """
        if os.path.exists(self.metadata_filename):
            os.remove(self.metadata_filename)
            logger.debug("Deleted metadata %s", self.metadata_filename)
        else:
            logger.debug("Attempted to delete non-existing metadata %s", self.metadata_filename)



