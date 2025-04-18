import logging
from typing import Dict, List
from datetime import datetime

from bson import ObjectId
from astropy.time import Time, TimeDelta
import spiceypy as spice


from src.SPICE.instruments.instrument import BaseInstrument
from src.SPICE.kernel_utils.kernel_management import BaseKernelManager
from src.simulation.filters import BaseFilter
from src.db.interface import Sessions


logger = logging.getLogger(__name__)

class DataFetchingEngine:
    """
    Orchestrate the data fetcher to downnload, parse and reproject data to our desired frame.
    """

    class InstrumetDataExtraction:
        ...

    class ExtractionState:
        ...


    def __init__(self, instruments: List[BaseInstrument], filter_object: BaseFilter, kernel_manager: BaseKernelManager):
        self.instruments = instruments
        self.filter = filter_object
        self.kernel_manager = kernel_manager
        self.threads = []



    def project_until_next_interval():
        """Projects collected points until the start of the next interval of interest of whatever instrument"""
        ...



    def check_threads(self):
        # Iterate in reverse to safely pop finished threads.
        for thread_id in range(len(self.threads) - 1, -1, -1):
            if self.threads[thread_id] is None:
                self.threads.pop(thread_id)

            elif not self.threads[thread_id].is_alive():
                self.threads[thread_id].join()


    def flush_SPICE(self):
        """
        Flush SPICE errors and reset the error state.
        """
        try:
            spice.reset()
            # TODO - Use some other random command which would be OK
            spice.utc2et(self.simulation_state.current_simulation_timestamp.utc.iso)
        except Exception as e:
            ...


    def create_metadata_record(self):
        '''Returns True when task already computed'''
        self.simulation_metadata_id = ObjectId()
        simulation_metadata = {
            "_id": self.simulation_metadata_id,  # Explicit ID
            "simulation_name": self._simulation_name,
            "simulation_attempt_name": self.simulation_name,
            "task_group_id": self.task_group_id,
            "start_time": self.start_time.utc.iso,
            "end_time": self.end_time.utc.iso,
            "last_logged_time": self.simulation_state.current_simulation_timestamp.utc.iso,
            "kernel_manager_min_time": (
                self.kernel_manager.min_loaded_time.utc.iso if self.kernel_manager.min_loaded_time else None
            ),
            "kernel_manager_max_time": (
                self.kernel_manager.max_loaded_time.utc.iso if self.kernel_manager.max_loaded_time else None
            ),
            "instruments": [instrument.name for instrument in self.instruments],
            "satellite_name": self.instruments[0].satellite_name,
            "filter_name": self.filter.name,
            "frame": self.kernel_manager.main_reference_frame,
            "created_at": datetime.datetime.now(),
            "finished": False,
        }
        return Sessions.prepare_simulation_metadata(simulation_metadata)


    def start_extraction(self):
        """
        Start the data extraction process. The main loop, controlling other components
        """
        ...

