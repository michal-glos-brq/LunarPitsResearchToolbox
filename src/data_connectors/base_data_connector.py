import logging
from typing import List, Dict, Tuple
from astropy.time import Time

logger = logging.getLogger(__name__)


class BaseDataProductManager:
    """
    Abstract class for managing an instrument's data products.
    Must be extended for instrument-specific parsing and data acquisition.
    """
    def __init__(self, instrument_name: str, data_source: str, global_interval: Tuple[Time, Time]):
        self.instrument_name = instrument_name
        self.data_source = data_source  # e.g., URL or local directory
        self.global_interval = global_interval  # (start_time, end_time)
        self.data_loaded: bool = False
        # For simplicity, hold preloaded segments as list of tuples:
        # (segment_start_et, segment_end_et, structured_array)
        self.segments: List[Tuple[float, float, np.ndarray]] = []

    def load_data(self, start_time: Time, end_time: Time) -> None:
        """
        Load (or download and parse) data from start_time to end_time into memory.
        Override this method with instrument-specific code.
        """
        logger.info(f"[{self.instrument_name}] Loading data from {start_time.utc.iso} to {end_time.utc.iso}")
        # --- IMPLEMENT instrument-specific loading here ---
        # For each file that intersects the time range:
        #    1. Download (if needed)
        #    2. Parse the file (e.g., using a format file such as LOLARDR.FMT)
        #    3. Append (seg_start_et, seg_end_et, data_array) to self.segments
        self.data_loaded = True  # Mark as loaded after prefetching

    def get_data_in_interval(self, start_et: float, end_et: float) -> List[Dict]:
        """
        Return a list (or array) of data points for the given ephemeris time interval.
        Each data point may be represented as a dictionary or structured array.
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        result = []
        for seg_start, seg_end, data_array in self.segments:
            # If the segment overlaps the requested interval:
            if seg_end < start_et or seg_start > end_et:
                continue
            # Assuming data_array has an 'et' field holding ephemeris time in seconds.
            mask = (data_array['et'] >= start_et) & (data_array['et'] < end_et)
            filtered = data_array[mask]
            result.extend(filtered.tolist())  # Converting to list of dicts
        return result

    def cleanup(self) -> None:
        """
        Free memory and remove temporary cached files if necessary.
        """
        logger.info(f"[{self.instrument_name}] Cleaning up loaded data.")
        self.segments.clear()
        self.data_loaded = False