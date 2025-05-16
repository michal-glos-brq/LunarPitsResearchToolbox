"""
============================================================
Area Filter implementation
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

import numpy as np
import spiceypy as spice

from .base_filter import BaseFilter
from src.global_config import LUNAR_RADIUS


class AreaFilter(BaseFilter):
    """This filter is used to filter out points that are not within a certain area on the lunar surface"""

    @classmethod
    def from_kwargs_and_kernel_manager(cls, _, **kwargs):
        """
        This method is used to create an AreaFilter instance from the given kernel manager and keyword arguments.
        It extracts the DSK filename from the kernel manager and uses it to initialize the filter.
        """
        if kwargs.get("min_lat") is None or kwargs.get("max_lat") is None:
            raise ValueError("min_lat and max_lat must be provided in kwargs")
        if kwargs.get("min_lon") is None or kwargs.get("max_lon") is None:
            raise ValueError("min_lon and max_lon must be provided in kwargs")

        return cls(kwargs["min_lat"], kwargs["max_lat"], kwargs["min_lon"], kwargs["max_lon"])

    def __init__(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float):
        """
        Defining an aread on the lunar surface in the simplest possible manner
        """
        # We assume latitude in <-90;90> and longitude in <0;360>
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

        self.lat_condition = lambda lat: self.min_lat <= lat <= self.max_lat
        self.lat_distance_function = lambda lat: min(abs(self.min_lat - lat), abs(self.max_lat - lat))

        if min_lon >= max_lon:
            self.lon_condition = lambda lon: lon >= self.min_lon or lon <= self.max_lon
            self.lon_distance_function = lambda lon: min(abs(self.min_lon - lon), abs(self.max_lon - lon))

        else:
            self.lon_condition = lambda lon: self.min_lon <= lon <= self.max_lon
            self.lon_distance_function = lambda lon: min(
                *(
                    [self.min_lon - lon, 360 - self.max_lon + lon]
                    if lon < self.min_lon
                    else [360 - lon + self.min_lon, lon - self.max_lon]
                )
            )

    @staticmethod
    def _name(**kwargs):
        """Obtain the filter unique name without instantiation"""
        return f"AreaFilter_{kwargs['min_lat']}_{kwargs['max_lat']}_{kwargs['min_lon']}_{kwargs['max_lon']}"

    @property
    def name(self) -> str:
        return f"AreaFilter_{self.min_lat}_{self.max_lat}_{self.min_lon}_{self.max_lon}"

    def rank_point(self, point: np.array) -> float:
        _, lon_rad, lat_rad = spice.reclat(point)
        lat_deg, lon_deg = np.degrees(lat_rad), np.degrees(lon_rad)

        # Check if inside the area
        lat_condition = self.lat_condition(lat_deg)
        lon_condition = self.lon_condition(lon_deg)

        if lat_condition and lon_condition:
            return 0  # Inside the area, no distance

        # Compute distances to the nearest boundary
        lat_dist = self.lat_distance_function(lat_deg)
        lon_dist = self.lon_distance_function(lon_deg)

        # Scale longitude by cos(latitude) to correct for spherical distortion
        lon_dist_scaled = lon_dist * np.cos(np.radians(lat_deg))

        # Compute final straight-line (Cartesian-like) distance
        return np.sqrt(lat_dist**2 + lon_dist_scaled**2) * (np.pi * LUNAR_RADIUS / 180)

