"""This file defines classes used to filter out projected points on the lunar surface"""

import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import spiceypy as spice
from scipy.spatial import cKDTree
from filelock import FileLock

from src.SPICE.kernel_utils.detailed_model import DetailedModelDSKKernel
from src.db.interface import Sessions
from src.global_config import LUNAR_RADIUS
from src.SPICE.config import DSK_KERNEL_LOCK_TIMEOUT, KERNEL_LOCK_POLL_INTERVAL

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

    @staticmethod
    @abstractmethod
    def _name(**kwargs) -> str:
        pass

    @abstractmethod
    def rank_point(self, point: np.array) -> float:
        pass

    @classmethod
    @abstractmethod
    def from_kwargs_and_kernel_manager(cls, kernel_manager, **kwargs):
        """
        This method is used to create a filter instance from the given kernel manager and keyword arguments.
        It extracts the DSK filename from the kernel manager and uses it to initialize the filter.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class PointFilter(BaseFilter):
    """This filter is used to filter out points that in a treshold distance to points on the lunar surface"""

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

    def __init__(self, hard_radius: float, dsk_filename: str):
        """
        hard_radius - radius around the point we want to caputure. Hard in the sense that no points within this treshold would be filtered out
        """
        self.dsk_filename = dsk_filename
        self._hard_radius = hard_radius
        with FileLock(dsk_filename + ".lock", timeout=DSK_KERNEL_LOCK_TIMEOUT, poll_interval=KERNEL_LOCK_POLL_INTERVAL):
            self._load_target_points()

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

    def _load_target_points(self):
        """Fetches crater points and converts lat/lon to Cartesian coordinates using DSK."""
        points = Sessions.get_all_pits_points()
        cartesian_points = DetailedModelDSKKernel.dsk_latlon_to_cartesian(
            points["latitude"], points["longitude"], self.dsk_filename
        )
        # Store computed points
        points["X"], points["Y"], points["Z"] = np.array(cartesian_points).T
        self._target_points = points[["X", "Y", "Z"]].values
        self._target_ids = np.arange(len(self._target_points))
        self.kd_tree = cKDTree(self._target_points)


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


class CompositeFilter(BaseFilter):
    """This filter is used to combine multiple filters"""

    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters

    @property
    def name(self) -> str:
        return f"CompositeFilter_{'_'.join([f.name for f in self.filters])}"

    def rank_point(self, point: np.array) -> float:
        return min([f.rank_point(point) for f in self.filters])
