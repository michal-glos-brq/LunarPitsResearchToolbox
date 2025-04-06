import logging

from src.SPICE.instruments.instrument import BaseInstrument
from src.SPICE.instruments.subinstruments import DivinerSubInstrument, SubInstrument, MiniRFSubInstrument
from src.SPICE.config import (
    DIVINER_SUBINSTRUMENTS,
    LOLA_INSTRUMENT_IDS,
    LRO_STR_ID,
    MINI_RF_CHANNELS,
    LROC_NAC_IDS,
    LROC_WAC_IDS,
)

logger = logging.getLogger(__name__)


class DivinerInstrument(BaseInstrument):
    # To correctly load initial state, min_time + 4 julian days ...

    STATIC_INSTRUMENT = False
    DYNAMIC_KERNEL_OFFSET_JD = 4.0
    name = "DIVINER"
    satellite_name = LRO_STR_ID
    # LRO spacecraft attitude frame
    frame = "LRO_SC_BUS"
    # Default FOV and height in case initialization is in time of failure for this instrument
    _fov_width = 610
    _height = 180

    _sub_instruments = None

    @property
    def sub_instruments(self):
        if self._sub_instruments is None:
            # Initialize sub_instruments only once
            self._sub_instruments = [
                DivinerSubInstrument(naif_id, _id, pixel_key) for naif_id, _id, pixel_key in DIVINER_SUBINSTRUMENTS
            ]
        return self._sub_instruments


class LolaInstrument(BaseInstrument):

    name = "LOLA"
    satellite_name = LRO_STR_ID
    # LRO spacecraft attitude frame
    frame = "LRO_SC_BUS"
    # Default FOV and height in case initialization is in time of failure for this instrument
    _fov_width = 1
    _height = 340

    _sub_instruments = None

    @property
    def sub_instruments(self):
        if self._sub_instruments is None:
            # Initialize sub_instruments only once
            self._sub_instruments = [SubInstrument(naif_id) for naif_id in LOLA_INSTRUMENT_IDS]
        return self._sub_instruments


class MiniRFInstrument(BaseInstrument):
    """https://pds-geosciences.wustl.edu/lro/lro-l-mrflro-4-cdr-v1/lromrf_0001/document/dp_sis/mrflro_dp_sis.pdf"""

    name = "MiniRF"
    satellite_name = LRO_STR_ID
    # LRO spacecraft attitude frame
    frame = "LRO_SC_BUS"
    # Default FOV and height in case initialization is in time of failure for this instrument
    _fov_width = 80
    _height = 300

    _sub_instruments = None

    @property
    def sub_instruments(self):
        if self._sub_instruments is None:
            # Initialize sub_instruments only once
            self._sub_instruments = [MiniRFSubInstrument(channel) for channel in MINI_RF_CHANNELS]
        return self._sub_instruments


class LROCNACInstrument(BaseInstrument):
    """
    LROC Narrow Angle Camera instrument.
    This instrument is defined by two subinstruments:
      - NAC-L (NAIF ID -85600)
      - NAC-R (NAIF ID -85610)
    Each subinstrument’s FOV and distortion parameters are retrieved
    via spice.getfov using the instrument kernel.
    """

    STATIC_INSTRUMENT = False
    name = "LROC_NAC"
    satellite_name = LRO_STR_ID
    # We choose a common reference frame (e.g., the LRO spacecraft bus frame)
    frame = "LRO_SC_BUS"
    # Default FOV and height in case initialization is in time of failure for this instrument
    _fov_width = 45
    _height = 220

    _sub_instruments = None

    @property
    def sub_instruments(self):
        if self._sub_instruments is None:
            # Initialize sub_instruments only once
            self._sub_instruments = [SubInstrument(naif_id) for naif_id in LROC_NAC_IDS]
        return self._sub_instruments


class LROCWACInstrument(BaseInstrument):
    """
    LROC Wide Angle Camera instrument.
    This instrument includes multiple filters, each with its own FOV,
    as defined in the instrument kernel. The NAIF IDs for the WAC filters are:
      - VIS: -85631, -85632, -85633, -85634, -85635
      - UV:  -85641, -85642
    They are collected here as subinstruments.
    """

    STATIC_INSTRUMENT = False
    name = "LROC_WAC"
    satellite_name = LRO_STR_ID
    frame = "LRO_SC_BUS"
    # Default FOV and height in case initialization is in time of failure for this instrument
    _fov_width = 250
    _height = 250

    _sub_instruments = None

    @property
    def sub_instruments(self):
        if self._sub_instruments is None:
            # Initialize sub_instruments only once
            self._sub_instruments = [SubInstrument(naif_id) for naif_id in LROC_WAC_IDS]
        return self._sub_instruments
