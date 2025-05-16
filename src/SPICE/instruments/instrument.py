"""
====================================================
SPICE Kernel Management Module
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project

This module defines the BaseInstrument class, which provides a standard
interface for lunar satellite instruments simulated using SPICE.

It includes:
    - Abstract fields for name, frame, spacecraft ID, and sub-instruments.
    - Computation of spacecraft position and velocity.
    - Vector projection (boresight and FOV bounds) onto the Moon's surface.
    - Geometry-based FOV intersection support.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import spiceypy as spice

from src.structures import ProjectionPoint
from src.SPICE.instruments.subinstruments import SubInstrument
from src.global_config import LUNAR_FRAME, LUNAR_FRAME
from src.SPICE.config import ABBERRATION_CORRECTION

logger = logging.getLogger(__name__)




class BaseInstrument(ABC):
    """
    Abstract base class representing a remote sensing instrument.

    Provides a general interface for instrument-specific geometry, including:
        - Boresight vector averaging from sub-instruments
        - Surface projection of boresight and bounds
        - FOV-to-point intersection metrics
        - Spacecraft position/velocity queries

    Subclasses must define:
        - `name`: Unique instrument identifier.
        - `frame`: SPICE instrument reference frame.
        - `satellite_name`: SPICE satellite identifier.
        - `sub_instruments`: List of SubInstrument instances.

    Supports both static and dynamic instruments.
    """

    STATIC_INSTRUMENT = False
    # Approximate values for estimation when there is no data yet and SPICE transformation could not be calculated
    _fov_width = 10000
    _height = 1000000
    # SPICE ID of the orbiting body (e.g., Moon)
    _orbiting_body = None

    def __init__(self):
        """
        Initialize default attributes and placeholders for geometric projections.

        Subclasses do not override this unless internal geometry model is different.
        """
        # Static boresight - aggregated boresight in static frame, used for instruments with dynamic boresight relatively to the satellite
        self._static_boresight, self._static_boresight_frame = None, None
        self._static_bounds, self._static_bounds_frame = None, None
        # self._boresights = None
        self._bounds_angle = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier of the instrument."""
        ...

    @property
    @abstractmethod
    def frame(self) -> str:
        """SPICE reference frame used by the instrument."""
        ...

    @property
    @abstractmethod
    def satellite_name(self) -> str:
        """SPICE spacecraft ID for retrieving position/velocity."""
        ...

    @property
    @abstractmethod
    def sub_instruments(self) -> List[SubInstrument]:
        """List of SubInstrument instances defining the geometry."""
        ...

    def calculate_spacecraft_position(self, et: float, frame: str) -> np.array:
        """
        Get spacecraft position at the specified ephemeris time.

        Parameters:
            et (float): Ephemeris time in seconds past J2000.
            frame (str): Target reference frame for the result.

        Returns:
            np.array: 3D Cartesian position vector in kilometers.
        """
        return spice.spkpos(self.satellite_name, et, frame, "NONE", self._orbiting_body)[0]

    def calculate_spacecraft_position_and_velocity(self, et: float, frame: str) -> Tuple[np.array]:
        """Returns spacecraft position and velocity in the given frame at the given time"""
        state = spice.spkezr(self.satellite_name, et, frame, "NONE", self._orbiting_body)[0]
        return state[:3], state[3:]

    def recalculate_bounds_to_boresight_distance(self, et: float = 0) -> float:
        """
        Recalculate the maximum angular distance from the boresight to the instrument bounds.

        Used to determine angular FOV spread for hit-testing projection intersections.

        Parameters:
            et (float): Ephemeris time to project at.

        Returns:
            float: Maximum distance in km between projected bounds and boresight.
        """
        self._last_recalculation_et = et
        projection = self.project_boresight(et).projection

        projected_bounds = np.stack([bnd.projection for bnd in self.project_bounds(et)])
        # projected_bounds = np.stack([self.project_vector(et, bnd).projection for bnd in self.bounds(et)])
        return np.linalg.norm(projected_bounds - projection, axis=1).max()


    def boresight(self, et: float = 0) -> np.array:
        """
        Get the current boresight vector at the given time.

        Combines static boresights of all subinstruments into a single vector,
        applying transformation if the instrument is not static.

        Parameters:
            et (float): Ephemeris time.

        Returns:
            np.array: 3D unit boresight vector.
        """
        if self._static_boresight is None:
            # Arbitrary choose the first frame - those frames are static relative to boresights
            self._static_boresight_frame = (
                self.sub_instruments[0].sub_instrument_frame if not self.STATIC_INSTRUMENT else self.frame
            )
            boresights = []
            for sub_instr in self.sub_instruments:
                boresights.append(sub_instr.transformed_boresight(self._static_boresight_frame, et=et))
            self._static_boresight = np.stack(boresights).mean(axis=0)

        if self.STATIC_INSTRUMENT:
            return self._static_boresight
        else:
            # Use first subinstrument to calculate the transformation
            return self.sub_instruments[0].transform_vector(self.frame, self._static_boresight, et)

    def bounds(self, et: float = 0) -> np.array:
        """
        Get the current FOV bounds vectors at the given time.

        Parameters:
            et (float): Ephemeris time.

        Returns:
            np.array: N×3 array of bound vectors.
        """
        if self._static_bounds is None:
            # Arbitrary choose the first frame - those frames are static relative to boresights
            self._static_bounds_frame = (
                self.sub_instruments[0].sub_instrument_frame if not self.STATIC_INSTRUMENT else self.frame
            )
            bounds = []
            for sub_instr in self.sub_instruments:
                bounds.append(sub_instr.transformed_bounds(self._static_bounds_frame, et=et))
            self._static_bounds = np.stack(bounds).reshape(-1, 3)
        if self.STATIC_INSTRUMENT:
            return self._static_bounds
        else:
            return np.stack(
                [self.sub_instruments[0].transform_vector(self.frame, bnd, et) for bnd in self._static_bounds]
            ).reshape(-1, 3)

    def project_vector(self, et, vector, use_ellipsoid: bool = False) -> np.array:
        """
        Project a direction vector from the spacecraft to the lunar surface.

        Uses SPICE's `sincpt` to find intersection point with the Moon's surface,
        either via DSK or ellipsoid approximation.

        Parameters:
            et (float): Ephemeris time.
            vector (np.array): Direction vector in `self.frame`.
            use_ellipsoid (bool): Whether to use the ellipsoid model.

        Returns:
            ProjectionPoint: Resulting intersection with the surface.
        """
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/sincpt_c.html
        boresight_point, boresight_trgepc, spacecraft_relative = spice.sincpt(
            "ELLIPSOID" if use_ellipsoid else "DSK/UNPRIORITIZED",
            "MOON",
            et,  # Time (just a number, the astro time)
            LUNAR_FRAME,
            ABBERRATION_CORRECTION,
            self.satellite_name,
            self.frame,
            vector,
        )
        return ProjectionPoint(et, boresight_point, boresight_trgepc, spacecraft_relative)

    def project_boresight(self, et) -> ProjectionPoint:
        """
        Project the instrument’s central boresight vector onto the lunar surface.

        Parameters:
            et (float): Ephemeris time.

        Returns:
            ProjectionPoint: Surface intersection of boresight.
        """
        return self.project_vector(et, self.boresight(et))

    def project_bounds(self, et) -> List[ProjectionPoint]:
        """
        Project all bound vectors of the instrument onto the lunar surface.

        Parameters:
            et (float): Ephemeris time.

        Returns:
            List[ProjectionPoint]: Surface intersections for each bound vector.
        """
        return [self.project_vector(et, bnd) for bnd in self.bounds(et)]
