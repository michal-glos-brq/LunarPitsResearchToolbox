"""This file defines classes used to filter out projected points on the lunar surface"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import spiceypy as spice
from scipy.spatial import cKDTree

from src.db.interface import Sessions
from src.global_config import LUNAR_RADIUS

logger = logging.getLogger(__name__)


class BaseFilter(ABC):

    _hard_radius = 0.0

    @property
    def hard_radius(self) -> float:
        # How far from out area or point of interest are we capturing the data
        return self._hard_radius

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def rank_point(self, point: np.array) -> float:
        pass


class PointFilter(BaseFilter):
    """This filter is used to filter out points that in a treshold distance to points on the lunar surface"""

    def __init__(self, hard_radius: float, dsk_filename: str):
        """
        hard_radius - radius around the point we want to caputure. Hard in the sense that no points within this treshold would be filtered out
        """
        self.dsk_filename = dsk_filename
        self._hard_radius = hard_radius
        self._load_target_points(dsk_filename)

    @property
    def name(self) -> str:
        return f"PointFilter_{self.hard_radius}"

    def rank_point(self, point: np.array) -> float:
        """Returns distance from our points of interest"""
        return self.kd_tree.query(point)[0]

    def _load_target_points(self, dsk_filename: str):
        """Fetches crater points and converts lat/lon to Cartesian coordinates using DSK."""
        points = Sessions.get_all_pits_points()
        lat_rad = np.radians(points["latitude"])
        lon_rad = np.radians(points["longitude"])

        # Load DSK file and get a valid handle
        dsk_handle = spice.dasopr(dsk_filename)  # Open the DSK file
        # Find the DSK segment descriptor
        dladsc = spice.dlabfs(dsk_handle)  # Get the first segment in the file

        cartesian_points = []
        for lat, lon in zip(lat_rad, lon_rad):
            # Convert lat/lon to a unit vector for the ray direction
            cartesians = np.array(spice.latrec(1.0, lon, lat))  # Ensure it's an array

            # We have to simulate observer above the ground, looking at [0,0,0], because interception could not be calculated from within
            _, spoint, found = spice.dskx02(dsk_handle, dladsc, cartesians * 10_000, (-1) * cartesians)

            if found:
                cartesian_points.append(spoint)
            else:
                logger.warning(f"No surface intercept found for lat: {lat}, lon: {lon}")
                cartesian_points.append([np.nan, np.nan, np.nan])  # Mark missing points

        # Close the DSK file
        spice.dascls(dsk_handle)

        # Store computed points
        points["X"], points["Y"], points["Z"] = np.array(cartesian_points).T
        self._target_points = points[["X", "Y", "Z"]].values
        self._target_ids = np.arange(len(self._target_points))
        self.kd_tree = cKDTree(self._target_points)


class AreaFilter(BaseFilter):
    """This filter is used to filter out points that are not within a certain area on the lunar surface"""

    def __init__(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float):
        """
        Defining an aread on the lunar surface in the simplest possible manner
        """
        # We assume latitude in <-90;90> and longitude in <0;360>
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

        self.lat_condition = lambda lat: self.min_lat <=  lat <= self.max_lat
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


class CompositeFilter(BaseFilter):
    """This filter is used to combine multiple filters"""

    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters

    @property
    def name(self) -> str:
        return f"CompositeFilter_{'_'.join([f.name for f in self.filters])}"

    def rank_point(self, point: np.array) -> float:
        return min([f.rank_point(point) for f in self.filters])
