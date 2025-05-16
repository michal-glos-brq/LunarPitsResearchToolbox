"""
====================================================
SPICE Kernel Management Module
====================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

from src.SPICE.instruments.instrument import BaseInstrument
from src.SPICE.instruments.subinstruments import SubInstrument
from src.SPICE.config import GRAIL_A_INSTRUMENTS, GRAIL_B_INSTRUMENTS


class GrailAInstrument(BaseInstrument):

    name = "GRAIL-A"
    satellite_name = "GRAIL-A"
    frame = "GRAIL-A_STA"

    _orbiting_body = "MOON"

    _sub_instruments = None

    @property
    def sub_instruments(self):
        if self._sub_instruments is None:
            self._sub_instruments = [SubInstrument(naif_id) for naif_id in GRAIL_A_INSTRUMENTS]
        return self._sub_instruments


class GrailBInstrument(BaseInstrument):
    name = "GRAIL-B"
    satellite_name = "GRAIL-B"
    frame = "GRAIL-B_STA"

    _orbiting_body = "MOON"


    _sub_instruments = None

    @property
    def sub_instruments(self):
        if self._sub_instruments is None:
            self._sub_instruments = [SubInstrument(naif_id) for naif_id in GRAIL_B_INSTRUMENTS]
        return self._sub_instruments

