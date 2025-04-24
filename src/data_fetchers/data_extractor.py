import logging
from contextlib import nullcontext
from typing import Dict, List, Optional
from datetime import datetime

from bson import ObjectId
from astropy.time import Time, TimeDelta
from tqdm import tqdm
import spiceypy as spice
from celery import Task

from src.SPICE.utils import SPICELog
from src.structures import IntervalManager
from src.SPICE.instruments.instrument import BaseInstrument
from src.SPICE.kernel_utils.kernel_management import BaseKernelManager
from src.data_fetchers.data_connectors.base_data_connector import BaseDataConnector
from src.data_fetchers.data_connectors.diviner_data_connector import DivinerDataConnector
from src.filters import BaseFilter
from src.db.interface import Sessions
from src.SPICE.utils import et2astropy_time
from src.data_fetchers.config import MONGO_UPLOAD_BATCH_SIZE
from src.global_config import SUPRESS_TQDM, TQDM_NCOLS, SPICE_DECIMAL_PRECISION

logger = logging.getLogger(__name__)


DATA_CONNECTOR_MAP = {
    "DIVINER": DivinerDataConnector,
}


class DataFetchingEngine:
    """
    Orchestrate the data fetcher to downnload, parse and reproject data to our desired frame.
    """

    class InstrumentState:

        def __init__(self, instrument: BaseInstrument):
            self.data_collection = Sessions.prepare_extraction_collections(
                instrument.name, instrument.timeseries, instrument.indices
            )
            self.total_data = 0
            self.instrument = instrument
            self.data = []
            self.data_done = []

    class ExtractionState:
        def __init__(
            self, kernel_manager: BaseKernelManager, instruments: List[BaseInstrument], filter_object: BaseFilter
        ):
            self.instruments = instruments
            self.filter = filter_object
            self.kernel_manager = kernel_manager
            self.last_reprojected_time_et = None
            self.last_interval_start_et = None

        def setup(
            self,
            interval_manager: IntervalManager,
            start_time: Time = None,
            end_time: Time = None,
            current_task: Optional[Task] = None,
            interactive_progress: bool = True,
            extraction_name: Optional[str] = None,
            supress_error_logs: bool = False,
            task_group_id: Optional[str] = None,
            retry_count: Optional[int] = None,
        ):
            self.simulation_metadata_id = ObjectId()
            self.start_time = self.kernel_manager.min_loaded_time if start_time is None else start_time
            self.end_time = self.kernel_manager.max_loaded_time if end_time is None else end_time
            self.retry_count = retry_count
            self.task_group_id = task_group_id
            self.current_task = current_task
            self.interval_manager = interval_manager
            self.supress_error_logs = supress_error_logs
            self.interactive_progress = interactive_progress
            self._extraction_name = extraction_name
            self.extraction_name = f"{extraction_name}_{retry_count}"

    def __init__(self, instruments: List[BaseInstrument], filter_object: BaseFilter, kernel_manager: BaseKernelManager):
        self.extraction_state = self.ExtractionState(kernel_manager, instruments, filter_object)
        self.instrument_states = {instr.name: self.InstrumentState(instr) for instr in instruments}
        self.threads = []

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
            spice.utc2et(self.extraction_state.start_time.utc.iso)
        except Exception as e:
            pass

    def create_metadata_record(self):
        """Returns True when task already computed"""
        simulation_metadata = {
            "_id": self.extraction_state.simulation_metadata_id,  # Explicit ID
            "extraction_name": self.extraction_state._extraction_name,
            "extraction_attempt_name": self.extraction_state.extraction_name,
            "task_group_id": self.extraction_state.task_group_id,
            "start_time": self.extraction_state.start_time.utc.iso,
            "end_time": self.extraction_state.end_time.utc.iso,
            "last_reprojected_time_et": self.extraction_state.last_reprojected_time_et,
            "last_interval_start_et": self.extraction_state.last_interval_start_et,
            "kernel_manager_min_time": (
                self.extraction_state.kernel_manager.min_loaded_time.utc.iso
                if self.extraction_state.kernel_manager.min_loaded_time
                else None
            ),
            "kernel_manager_max_time": (
                self.extraction_state.kernel_manager.max_loaded_time.utc.iso
                if self.extraction_state.kernel_manager.max_loaded_time
                else None
            ),
            "instruments": list(self.extraction_state.instruments.keys()),
            "satellite_name": list(self.extraction_state.instruments.values())[0].satellite_name,
            "filter_name": self.extraction_state.filter.name,
            "frame": self.extraction_state.kernel_manager.main_reference_frame,
            "created_at": datetime.datetime.now(),
            "finished": False,
        }
        return Sessions.prepare_extraction_metadata(simulation_metadata)

    def iteration_housekeeping(self):
        for instr_state in self.instrument_states.values():
            if len(instr_state.data_done) > MONGO_UPLOAD_BATCH_SIZE:
                # If we have more than MONGO_UPLOAD_BATCH_SIZE, we can push to MongoDB
                thread = Sessions.start_background_batch_insert(instr_state.data_done)
                instr_state.data_done = []
                self.threads.append(thread)
        self.check_threads()


    def process_data(self):

        def get_oldest_data_entry():
            """
            Get the oldest data entry from all instruments.
            """
            oldest_entry = None
            for instrument_state in self.instrument_states.values():
                if instrument_state.data and instrument_state.data[0]["et"] < self.last_interval_start_et:
                    if oldest_entry is None or instrument_state.data[0]["et"] < oldest_entry["et"]:
                        oldest_entry = (instrument_state.instrument.name, instrument_state.data.pop(0))
            return oldest_entry

        while dato := get_oldest_data_entry():
            instrument_name, data_entry = dato
            try:
                reprojected_data = self.data_connectors[instrument_name].process_data_entry(
                    data_entry, self.instrument_states[instrument_name].instrument, self.extraction_state.filter
                )

                if reprojected_data:
                    self.instrument_states[instrument_name].data_done.append(reprojected_data)
            except Exception as e:
                SPICELog.log_spice_exception(e, context=f"Error processing data entry for instrument {instrument_name}")
                self.flush_SPICE()

    def setup_data_connectors(self) -> Dict[str, BaseDataConnector]:
        connectors = {}
        for name, interval_list in self.extraction_state.interval_manager.intervals.items():
            connectors[name] = DATA_CONNECTOR_MAP[name](self.extraction_state.kernel_manager, interval_list)
        return connectors

    def start_extraction(
        self,
        interval_manager: IntervalManager,
        start_time: Time = None,
        end_time: Time = None,
        current_task: Optional[Task] = None,
        interactive_progress: bool = True,
        extraction_name: Optional[str] = None,
        supress_error_logs: bool = False,
        task_group_id: Optional[str] = None,
        retry_count: Optional[int] = None,
    ):
        """
        Start the data extraction process. The main loop, controlling other components
        """
        self.extraction_state.setup(
            interval_manager,
            start_time,
            end_time,
            current_task,
            interactive_progress,
            extraction_name,
            supress_error_logs,
            task_group_id,
            retry_count,
        )

        self.extraction_state.kernel_manager.step(self.extraction_state.start_time)
        self.data_connectors = self.setup_data_connectors()

        # Do the mongo DB
        pbar = (
            tqdm(total=len(self.extraction_state.interval_manager), disable=SUPRESS_TQDM, ncols=TQDM_NCOLS)
            if interactive_progress
            else nullcontext()
        )

        # Log the simulation state into Mongo
        task_already_finished = self.create_metadata_record()
        if task_already_finished:
            if current_task is not None:
                current_task.update_state(
                    state="SUCCESS", meta={"result": "Task already finished, no computation to do."}
                )
            return

        SPICELog.interactive_progress = interactive_progress
        SPICELog.supress_output = supress_error_logs

        self.data_collections = {
            instr.name: Sessions.prepare_extraction_collections(instr.name, instr.timeseries, instr.indices)
            for instr in self.extraction_state.instruments.values()
        }

        # Iterate through the Interval Queu
        with pbar:
            while True:
                try:
                    instrument_interval_tuple = self.extraction_state.interval_manager.next_interval()
                    if instrument_interval_tuple is None:
                        break
                    instrument_name, interval = instrument_interval_tuple
                    self.last_interval_start_et = interval.start_et

                    new_data = self.data_connectors[instrument_name].read_interval(interval)
                    if new_data:
                        # Enhancing metadata for tracebility
                        for dato in new_data:
                            dato["meta"] = {
                                "_extraction_name": self.extraction_state._extraction_name,
                                "extraction_name": self.extraction_state.extraction_name,
                                "task_group_id": self.extraction_state.task_group_id,
                            }
                        self.instrument_states[instrument_name].data.extend(new_data)
                        self.instrument_states[instrument_name].total_data += len(new_data)

                    self.process_data()
                except Exception as e:
                    SPICELog.log_spice_exception(
                        e, context=f"Error processing interval {interval} for instrument {instrument_name}"
                    )
                    self.flush_SPICE()
                finally:
                    self.check_threads()
                    self.iteration_housekeeping()
                    pbar.update(1)

        self.last_interval_start_et = spice.utc2et(self.extraction_state.end_time.utc.iso)
        self.process_data()

        # Await all threads.
        for thread in self.threads:
            thread.join()

        Sessions.process_failed_inserts()

        update_thread = Sessions.start_background_update_extraction_metadata(
            self.extraction_state.simulation_metadata_id,
            et2astropy_time(self.extraction_state.end_time).datetime,
            finished=True,
            metadata={instr_state.name: instr_state.total_data for instr_state in self.instrument_states.values()},
        )
        self.threads.append(update_thread)

        self.check_threads()

        if current_task is not None:
            # Update the task state to SUCCESS
            current_task.update_state(
                state="SUCCESS", meta={"result": "Task already finished, no computation to do."}
            )
