import logging

import spiceypy as spice

from src.SPICE.instruments.instrument import BaseInstrument
from src.SPICE.instruments.subinstruments import FrameCorrectedSubInstrument

logger = logging.getLogger(__name__)


LRO_STR_ID = "LUNAR RECONNAISSANCE ORBITER"


DIVINER_INSTRUMENT_IDS = [-85211, -85212, -85213, -85214, -85215, -85216, -85221, -85222, -85223]


class DivinerInstrument(BaseInstrument):

    name = "DIVINER"
    satellite_name = LRO_STR_ID
    frame = "LRO_DLRE"
    # Instruments are normalized into the same frame
    sub_instruments = [FrameCorrectedSubInstrument(naif_id, "LRO_DLRE") for naif_id in DIVINER_INSTRUMENT_IDS]


# class DIVINERInstrument(Instrument):
#     sweep_iterator_class = LROSweepIterator

#     name = "DIVINER"
#     instrument_ids = DIVINER_INSTRUMENT_IDS
#     # Frame of the instrument
#     frame = LRO_DIVINER_FRAME_STR_ID
#     satellite_frame = LRO_STR_ID

#     # Until succesfull Lunara projection
#     offset_days = 6.3552

#     # Distance on the Lunar surface between the middle of all suinstruments and furthest subinstrument (4.3 Km measured from low sample)
#     subinstrumen_offset = 6
#     # Distance from projected boresight to bound
#     fov_offset = 3

#     # Distance of fov from pit treshold
#     distance_tolerance = 12 * QUERY_RADIUS_MULTIPLIER


# class LOLAInstrument(Instrument):
#     sweep_iterator_class = LROSweepIterator

#     name = "LOLA"
#     instrument_ids = LOLA_INSTRUMENT_IDS
#     # Frame of the instrument
#     frame = LRO_LOLA_FRAME_STR_ID
#     satellite_frame = LRO_STR_ID
#     # Until succesfull Lunara projection
#     offset_days = 6.3552

#     ### Offsets calculating in additional tolerance - include spots sensed by instrument boiunds too, compute only the avarage boresight
#     # Distance on the Lunar surface between the middle of all suinstruments and furthest subinstrument (4.3 Km measured from low sample)
#     subinstrumen_offset = 1
#     # Distance from projected boresight to bound
#     fov_offset = 1

#     ### Distance tolerance - how far data from our points of interest are we interested int
#     distance_tolerance = 12 * QUERY_RADIUS_MULTIPLIER


# class MiniRFInstrument(Instrument):
#     sweep_iterator_class = LROSweepIterator

#     name = "MiniRF"
#     instrument_ids = MINIRF_INSTRUMENT_IDS
#     # Frame of the instrument
#     # frame = LRO_MINIRF_FRAME_STR_ID
#     frame = "LRO_SC_BUS"
#     satellite_frame = LRO_STR_ID
#     # Until succesfull Lunara projection
#     offset_days = 6.3552

#     # Distance on the Lunar surface between the middle of all suinstruments and furthest subinstrument (4.3 Km measured from low sample)
#     subinstrumen_offset = 6
#     # Distance from projected boresight to bound
#     fov_offset = 3

#     # Distance of fov from pit treshold
#     distance_tolerance = 12 * QUERY_RADIUS_MULTIPLIER

#     def instantiate_subinstruments(self):
#         """Missing instrument kernel, instrument metadata has to be configured manually"""
#         self.sub_instruments[self.instrument_ids[0]] = Instrument.SubInstrument(
#             _id=self.instrument_ids[0],
#             frame=self.frame,
#             boresight=spice.mxv(
#                 spice.pxform(LRO_MINIRF_FRAME_STR_ID, self.frame, self.current_simulation_timestamp_et), [0, 0, 1]
#             ),
#             # TODO: Use this: https://naif.jpl.nasa.gov/naif/data_lunar.html
#             bounds=...,
#         )


# class LROCWACInstrument(Instrument):
#     sweep_iterator_class = LROSweepIterator

#     name = "LROC WAC"

#     frame = ...
#     satellite_frame = LRO_STR_ID
#     # Until succesfull Lunar projection
#     offset_days = 6.3552

#     # FOV attributes TBD
#     subinstrumen_offset = ...
#     fov_offset = ...
#     distance_tolerance = ...


# class LROCNACInstrument(Instrument):
#     sweep_iterator_class = LROSweepIterator

#     name = "LROC NAC"

#     frame = ...
#     satellite_frame = LRO_STR_ID
#     # Until succesfull Lunar projection
#     offset_days = 6.3552

#     # FOV attributes TBD
#     subinstrumen_offset = ...
#     fov_offset = ...
#     distance_tolerance = ...
