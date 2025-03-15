"""This file implements Base class for instruments. Instruments themselves do not manage any SPICE kernels directly, it assumes all required kernels are loaded (otherwise an exception is throwed)"""

import sys
import logging
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
import spiceypy as spice

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


sys.path.insert(0, "/".join(__file__.split("/")[:-4]))
from src.config import (
    MOON_STR_ID,
    ABBERRATION_CORRECTION,
    LUNAR_FRAME,
)
from src.instruments.subinstruments import SubInstrument


class BaseInstrument(ABC):
    """This class serves the purpose of defining all the configurable attributes of instruments"""

    _bounds = None
    _boresight = None

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

    ### Boresight and bounds assume subinstruments are normalized into the same frame
    @property
    def boresight(self) -> np.array:
        """Boresight vector for the instrument"""
        if self._boresight is None:
            boresights = []
            for sub_instr in self.sub_instruments:
                if sub_instr.sub_instrument_frame != self.frame:
                    boresights.append(sub_instr.transformed_boresight(self.frame))
                else:
                    boresights.append(sub_instr.boresight)
            self._boresight = np.stack(boresights).mean(axis=0)
        return self._boresight

    @property
    def bounds(self) -> np.array:
        """Bounds of the instrument"""
        # TODO: Make it more efficient, this contains many redundant points
        if self._bounds is None:
            bounds = []
            for sub_instr in self.sub_instruments:
                if sub_instr.sub_instrument_frame != self.frame:
                    bounds.append(sub_instr.transformed_bounds(self.frame))
                else:
                    bounds.append(sub_instr.bounds)
            self._bounds = np.stack(bounds)
        return self._bounds

    def project_vector(self, et, vector) -> np.array:
        """
        Projects a vector pointing from satellite_name, onto the lunar surface. Return intersection of surface and vector in cartesian coordinates

        Takes vector in self.frame and projects it - expects subinstrument vector to be tranformed into the self.frame already
        """
        return spice.sincpt(
            "DSK/UNPRIORITIZED",
            MOON_STR_ID,
            et,  # Time (just a number, the astro time)
            LUNAR_FRAME,
            ABBERRATION_CORRECTION,
            self.satellite_name,
            self.frame,
            vector,
        )

    def project_mean_boresight(self, et) -> Dict[str, Dict]:
        """
        Project subinstrument-mean boresight onto lunar surface
        """
        boresight_point, boresight_trgepc, _ = self.project_vector(et, self.boresight)
        return {"et": et, "boresight": boresight_point, "boresight_trgepc": boresight_trgepc}

    def project_all_bounds(self, et) -> Dict[int, Dict]:
        """
        Project sub-instrument bounds to the lunar surface
        """
        # Here, we store ProjectedPoint for each subinstrument
        boresights = {}

        for sub_instrument in self.sub_instruments.values():
            boresight_point, boresight_trgepc, _ = self.project_vector(et, sub_instrument.boresight)
            boresights[sub_instrument._id] = {
                "et": et,
                "boresight": boresight_point,
                "boresight_trgepc": boresight_trgepc,
            }
        return boresights


    # TODO: Some bounds approximation, recalculation of bounds and distances from boresight and stuff ...

