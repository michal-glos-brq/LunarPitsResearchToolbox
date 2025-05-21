"""
============================================================
Filter by proximity to arbitrary defined poitns
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""


import numpy as np
from scipy.spatial import cKDTree

from .base_filter import BaseFilter


class PointFilter(BaseFilter):

    def __init__(self, hard_radius: float, points: np.array):
        """
        hard_radius - radius around the point we want to caputure. Hard in the sense that no points within this treshold would be filtered out
        dsk_filename - DSK file name
        """
        self._hard_radius = hard_radius
        self._target_points = points
        self.kd_tree = cKDTree(self._target_points)

    @classmethod
    def from_kwargs_and_kernel_manager(cls, _, **kwargs):
        """
        This method is used to create a PointFilter instance from the given kernel manager and keyword arguments.
        It extracts the DSK filename from the kernel manager and uses it to initialize the filter.
        """
        return cls(**kwargs)

    @property
    def name(self) -> str:
        return f"PointFilter_{self.hard_radius}"

    @staticmethod
    def _name(**kwargs):
        """Obtain the filter unique name without instantiation"""
        return f"PointFilter_{kwargs['hard_radius']}"

    def rank_point(self, point: np.array) -> float:
        """Returns distance from our points of interest"""
        return self.kd_tree.query(point)[0]

    def point_pass(self, point: np.array):
        """Check if the point is within the hard radius"""
        return self.kd_tree.query(point)[0] <= self.hard_radius
