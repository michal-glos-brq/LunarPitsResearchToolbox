from src.SPICE.instruments.instrument import BaseInstrument
from src.SPICE.instruments.subinstruments import SubInstrument
from src.SPICE.config import GRAIL_A_INSTRUMENTS, GRAIL_B_INSTRUMENTS


class GrailAInstrument(BaseInstrument):
    name = "GRAIL-A"
    satellite_name = "GRAIL-A"
    frame = "GRAIL-A_STA"
    sub_instruments = [SubInstrument(naif_id) for naif_id in GRAIL_A_INSTRUMENTS]


class GrailBInstrument(BaseInstrument):
    name = "GRAIL-B"
    satellite_name = "GRAIL-B"
    frame = "GRAIL-B_STA"
    sub_instruments = [SubInstrument(naif_id) for naif_id in GRAIL_B_INSTRUMENTS]
