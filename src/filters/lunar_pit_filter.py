"""
============================================================
Filter based on proximity to known lunar pits
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

import numpy as np
from filelock import FileLock

from .point_filter import PointFilter
from src.db.interface import Sessions
from src.SPICE.kernel_utils.detailed_model import DetailedModelDSKKernel
from src.SPICE.config import DSK_KERNEL_LOCK_TIMEOUT, KERNEL_LOCK_POLL_INTERVAL


class LunarPitFilter(PointFilter):
    """This filter is used to filter out points that are not within a treshold distance to points on the lunar surface"""

    def __init__(self, hard_radius: float, dsk_filename: str):
        """
        hard_radius - radius around the point we want to caputure. Hard in the sense that no points within this treshold would be filtered out
        dsk_filename - DSK file name
        """
        self.dsk_filename = dsk_filename
        with FileLock(dsk_filename + ".lock", timeout=DSK_KERNEL_LOCK_TIMEOUT, poll_interval=KERNEL_LOCK_POLL_INTERVAL):
            super().__init__(hard_radius, self._load_target_points())

    @property
    def name(self):
        """Obtain the filter unique name"""
        return f"LunarPitFilter_{self.hard_radius}"

    @classmethod
    def from_kwargs_and_kernel_manager(cls, kernel_manager, **kwargs):
        """
        This method is used to create a PointFilter instance from the given kernel manager and keyword arguments.
        It extracts the DSK filename from the kernel manager and uses it to initialize the filter.
        """
        dsk_filename = kernel_manager.static_kernels["dsk"][0].filename
        if kwargs.get("hard_radius") is None:
            raise ValueError("hard_radius must be provided in kwargs")
        return cls(kwargs["hard_radius"], dsk_filename)

    def _load_target_points(self):
        """Fetches crater points and converts lat/lon to Cartesian coordinates using DSK."""
        points = Sessions.get_all_pits_points()
        cartesian_points = DetailedModelDSKKernel.dsk_latlon_to_cartesian(
            points["latitude"], points["longitude"], self.dsk_filename
        )
        # Store computed points
        points["X"], points["Y"], points["Z"] = np.array(cartesian_points).T
        return points[["X", "Y", "Z"]].values
