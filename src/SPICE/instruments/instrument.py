"""This file implements Base class for instruments. Instruments themselves do not manage any SPICE kernels directly, it assumes all required kernels are loaded (otherwise an exception is throwed)"""

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
    """This class serves the purpose of defining all the configurable attributes of instruments"""

    STATIC_INSTRUMENT = False
    DYNAMIC_KERNEL_OFFSET_JD = 0
    # Use arbitrary absurd default values, until it's possible to be computed
    _fov_width = 10000
    _height = 1000000
    _orbiting_body = "MOON"

    def __init__(self):
        # Static boresight - aggregated boresight in static frame, used for instruments with dynamic boresight relatively to the satellite
        self._static_boresight, self._static_boresight_frame = None, None
        self._static_bounds, self._static_bounds_frame = None, None
        # self._boresights = None
        self._bounds_angle = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Instrument name, string ID"""
        ...

    @property
    @abstractmethod
    def frame(self) -> str:
        """Frame of the instrument"""
        ...

    @property
    @abstractmethod
    def satellite_name(self) -> str:
        """Satellite name, string ID (SPICE kernel ID) - spacecraft position"""
        ...

    @property
    @abstractmethod
    def sub_instruments(self) -> List[SubInstrument]:
        """List of subinstrument objects"""
        ...

    def calculate_spacecraft_position(self, et: float, frame: str) -> np.array:
        """Returns spacecraft position in the given frame at the given time"""
        return spice.spkpos(self.satellite_name, et, frame, "NONE", self._orbiting_body)[0]

    def calculate_spacecraft_position_and_velocity(self, et: float, frame: str) -> Tuple[np.array]:
        """Returns spacecraft position and velocity in the given frame at the given time"""
        state = spice.spkezr(self.satellite_name, et, frame, "NONE", self._orbiting_body)[0]
        return state[:3], state[3:]

    def recalculate_bounds_to_boresight_distance(self, et: float = 0) -> np.array:
        """
        Recalculates distance from boresight to bounds

        So we can add this to the distance from point of interest and with point of interest and projected boresight -
            we can tell wether instrument FOV is intersecting the point of interest
        """
        self._last_recalculation_et = et
        projection = self.project_boresight(et).projection

        projected_bounds = np.stack([bnd.projection for bnd in self.project_bounds(et)])
        # projected_bounds = np.stack([self.project_vector(et, bnd).projection for bnd in self.bounds(et)])
        return np.linalg.norm(projected_bounds - projection, axis=1).max()

    ### Boresight and bounds assume subinstruments are normalized into the same frame
    def boresight(self, et: float = 0) -> np.array:
        """Boresight vector for the instrument"""
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
        """Bounds of the instrument"""
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

    def project_vector(self, et, vector) -> np.array:
        """
        Projects a vector pointing from satellite_name, onto the lunar surface. Return intersection of surface and vector in cartesian coordinates

        Takes vector in self.frame and projects it - expects subinstrument vector to be tranformed into the self.frame already
        """
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/sincpt_c.html
        boresight_point, boresight_trgepc, spacecraft_relative = spice.sincpt(
            "DSK/UNPRIORITIZED",
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
        Project subinstrument-mean boresight onto lunar surface
        """
        return self.project_vector(et, self.boresight(et))

    def project_bounds(self, et) -> List[ProjectionPoint]:
        """
        Projects bounds onto lunar surface
        """
        return [self.project_vector(et, bnd) for bnd in self.bounds(et)]
