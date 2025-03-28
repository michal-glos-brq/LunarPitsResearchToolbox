"""This file implements Base class for instruments. Instruments themselves do not manage any SPICE kernels directly, it assumes all required kernels are loaded (otherwise an exception is throwed)"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Union
from functools import lru_cache, partial

import numpy as np
import spiceypy as spice

from src.SPICE.instruments.subinstruments import SubInstrument
from src.global_config import LUNAR_FRAME, LUNAR_FRAME
from src.SPICE.config import ABBERRATION_CORRECTION, BOUNDS_TO_BORESIGHT_BUFFER_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



# TODO: Some bounds approximation, recalculation of bounds and distances from boresight and stuff ...

@lru_cache(maxsize=1024)
def spacecraft_position(spacecraft_name: str, et: float, frame: str) -> np.array:
    """Returns spacecraft position in the given frame at the given time"""
    return spice.spkpos(spacecraft_name, et, frame, ABBERRATION_CORRECTION, "MOON")[0]


@dataclass
class ProjectionPoint:
    et: float
    projection: np.array
    projection_trgepc: float
    spacecraft_relative: np.array


class BaseInstrument(ABC):
    """This class serves the purpose of defining all the configurable attributes of instruments"""

    STATIC_INSTRUMENT = False

    def __init__(self):
    # This should probably be bounds, but those are bounds for a single subinstrument
        self._bound = None
        self._bounds = None
        self._boresight = None
        self._boresights = None
        self._bounds_angle = None

        self._boresight_to_bound_distance_buffer = np.zeros((BOUNDS_TO_BORESIGHT_BUFFER_LEN))
        self._boresight_to_bound_distance_index = 0
        self._boresight_to_bound_distance = 0
        self._last_recalculation_et = None

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


    def bounds_to_boresight_distance(self, et: float) -> float:
        """Distance from boresight to bounds"""
        if self._last_recalculation_et != et:
            self.recalculate_bounds_to_boresight_distance(et)
        return self._boresight_to_bound_distance

    def recalculate_bounds_to_boresight_distance(self, et: float = 0) -> np.array:
        """
        Recalculates distance from boresight to bounds
        
        So we can add this to the distance from point of interest and with point of interest and projected boresight -
            we can tell wether instrument FOV is intersecting the point of interest
        """
        self._last_recalculation_et = et
        projection = self.project_boresight(et).projection

        projected_bounds = np.stack([self.project_vector(et, bnd).projection for bnd in self.bounds(et)])
        distances = np.linalg.norm(projected_bounds - projection, axis=1)

        self._boresight_to_bound_distance_buffer[self._boresight_to_bound_distance_index] = distances.max()
        self._boresight_to_bound_distance_index = (self._boresight_to_bound_distance_index + 1) % BOUNDS_TO_BORESIGHT_BUFFER_LEN
        self._boresight_to_bound_distance = self._boresight_to_bound_distance_buffer.max()



    # def boresights(self, et: float = 0) -> np.array:
    #     """Not transformed Boresight vectors for the instrument"""
    #     return np.stack([sub_instr.boresight for sub_instr in self.sub_instruments])

    ### Boresight and bounds assume subinstruments are normalized into the same frame
    def boresight(self, et: float = 0) -> np.array:
        """Boresight vector for the instrument"""
        if self._boresight is None and not self.STATIC_INSTRUMENT:
            boresights = []
            for sub_instr in self.sub_instruments:
                boresights.append(sub_instr.transformed_boresight(self.frame, et=et))
            self._boresight = np.stack(boresights).mean(axis=0)
        return self._boresight


    def bounds(self, et: float = 0) -> np.array:
        """Bounds of the instrument"""
        if self._bounds is None and not self.STATIC_INSTRUMENT:
            bounds = []
            for sub_instr in self.sub_instruments:
                bounds.append(sub_instr.transformed_bounds(self.frame, et=et))
            self._bounds = np.stack(bounds).reshape(-1, 3)
        return self._bounds

    # def pareto_bounds(self) -> np.array:
    #     """Bounds of the instrument"""
    #     raise NotImplemented("Stuck to using redundant self.bounds")

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
        # Spacecraft relative is point relative to the point of interception. Add those 2 vectors to get position of spacecraft in the self.frame reference
        # return {"et": et, "projection": boresight_point, "projection_trgepc": boresight_trgepc, 'spacecraft_relative': spacecraft_relative}
