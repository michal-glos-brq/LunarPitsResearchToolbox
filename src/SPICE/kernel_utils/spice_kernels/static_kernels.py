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

from src.SPICE.kernel_utils.spice_kernels.base_static_kernel import BaseKernel

logger = logging.getLogger(__name__)



class AutoUpdateKernel(BaseKernel):
    """Kernel that automatically picks the newest file from a remote folder, based on kernel name, assuming datetime is present."""

    url: str
    filename: str

    def __init__(self, folder_url: str, folder_path: str, regex: str, keep_kernel: bool = True):
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
    Kernel with a filename and metadata filename with lbl suffix
    """

    def __init__(
        self, url: str, filename: str, metadata_url: str, metadata_filename: str, keep_kernel: bool = True
    ) -> None:
        BaseKernel.__init__(self, url, filename, keep_kernel=keep_kernel)
        self.metadata_url = metadata_url
        self.metadata_filename = metadata_filename

    @property
    def metadata_exists(self) -> bool:
        """Check if the metadata file exists on disk"""
        return os.path.exists(self.metadata_filename)

    def download_metadata(self) -> None:
        """Download the metadata file from the internet"""
        if not self.metadata_exists:
            logger.debug("Downloading metadata: %s", self.metadata_filename)
            self.download_file(self.metadata_url, self.metadata_filename)
        else:
            logger.debug("Metadata already exists: %s", self.metadata_filename)

    def delete_metadata(self) -> None:
        if os.path.exists(self.metadata_filename):
            os.remove(self.metadata_filename)
            logger.debug("Deleted metadata %s", self.metadata_filename)
        else:
            logger.debug("Attempted to delete non-existing metadata %s", self.metadata_filename)



