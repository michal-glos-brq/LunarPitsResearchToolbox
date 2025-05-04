import logging
from contextlib import nullcontext
from typing import Dict, List, Optional
from datetime import datetime

from bson import ObjectId
from astropy.time import Time
from tqdm import tqdm
import spiceypy as spice
from spiceypy import NotFoundError
from celery import Task

from src.SPICE.utils import SPICELog
from src.structures import IntervalManager
from src.SPICE.instruments.instrument import BaseInstrument
from src.SPICE.kernel_utils.kernel_management import BaseKernelManager
from src.data_fetchers.data_connectors.base_data_connector import BaseDataConnector
from src.data_fetchers.data_connectors.diviner_data_connector import DivinerDataConnector
from src.data_fetchers.data_connectors.lola_data_connector import LOLADataConnector
from src.data_fetchers.data_connectors.mini_rf_data_connector import MiniRFDataConnector
from src.filters import BaseFilter
from src.db.interface import Sessions
from src.data_fetchers.config import MONGO_UPLOAD_BATCH_SIZE
from src.global_config import SUPRESS_TQDM, TQDM_NCOLS
from src.SPICE.utils import et2astropy_time


logger = logging.getLogger(__name__)


DATA_CONNECTOR_MAP = {
    DivinerDataConnector.name: DivinerDataConnector,
    MiniRFDataConnector.name: MiniRFDataConnector,
    LOLADataConnector.name: LOLADataConnector,
}


class DataFetchingEngine:
    """
    Orchestrate the data fetcher to downnload, parse and reproject data to our desired frame.
    """

    class InstrumentState:

        def __init__(self, instrument: BaseInstrument, instrument_connector: BaseDataConnector):
            self.data_collection = Sessions.prepare_extraction_collections(
                instrument.name, instrument_connector.timeseries, instrument_connector.indices
            )
            self.instrument = instrument
            # Here we collect data filtered just by time interval
            self.data = []
            self.total_data = 0
            # Here we collect data reprojected and filtered with filter object, ready for database push
            self.reprojected_data = []
            self.total_reprojected_data = 0

    class ExtractionState:
        def __init__(
            self,
            kernel_manager: BaseKernelManager,
            instruments: List[BaseInstrument],
            filter_object: BaseFilter,
            custom_filter_objects: Dict[str, BaseFilter] = {},
        ):
            self.instruments = instruments
            self.filter = filter_object
            self.custom_filter_objects = custom_filter_objects
            self.kernel_manager = kernel_manager
            # There might be instrument overlaps, hence we reproject data until the past interval start
            self.last_reprojected_time_et = spice.utc2et(kernel_manager.min_loaded_time.utc.iso)
            self.last_interval_start_et = spice.utc2et(kernel_manager.min_loaded_time.utc.iso)

        def get_filter(self, instrument_name: Optional[str] = None) -> BaseFilter:
            """
            Get the filter object for the given instrument name.
            """
            return self.custom_filter_objects.get(instrument_name, self.filter)

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
            self._extraction_name = extraction_name if extraction_name else datetime.now().strftime("%Y%m%d_%H%M%S")
            self.extraction_name = f"{extraction_name}_{retry_count}" if retry_count is not None else extraction_name

    def __init__(
        self,
        instruments: List[BaseInstrument],
        filter_object: BaseFilter,
        kernel_manager: BaseKernelManager,
        custom_filter_objects: Dict[str, BaseFilter] = {},
    ):
        self.extraction_state = self.ExtractionState(kernel_manager, instruments, filter_object, custom_filter_objects)
        self.threads = []

    def check_threads(self):
        # Iterate in reverse to safely pop finished threads.
        for thread_id in range(len(self.threads) - 1, -1, -1):
            if self.threads[thread_id] is None:
                self.threads.pop(thread_id)
            elif not self.threads[thread_id].is_alive():
                thread = self.threads.pop(thread_id)
                thread.join()

    def flush_SPICE(self):
        """
        Flush SPICE errors and reset the error state.
        """
        try:
            spice.reset()
            spice.utc2et(self.extraction_state.start_time.iso)
        except Exception as e:
            pass

    def create_metadata_record(self):
        """Returns True when task already computed"""
        simulation_metadata = {
            "_id": self.extraction_state.simulation_metadata_id,  # Explicit ID
            # Raw name without retry count
            "extraction_name": self.extraction_state._extraction_name,
            # Name with retry count
            "extraction_attempt_name": self.extraction_state.extraction_name,
            # Task attribution to particular task dispatching event (running extractor dispatcher)
            "task_group_id": self.extraction_state.task_group_id,
            "start_time": self.extraction_state.start_time.iso,
            "end_time": self.extraction_state.end_time.iso,
            "last_reprojected_time_et": self.extraction_state.last_reprojected_time_et,
            "last_interval_start_et": self.extraction_state.last_interval_start_et,
            "kernel_manager_min_time": (
                self.extraction_state.kernel_manager.min_loaded_time.iso
                if self.extraction_state.kernel_manager.min_loaded_time
                else None
            ),
            "kernel_manager_max_time": (
                self.extraction_state.kernel_manager.max_loaded_time.iso
                if self.extraction_state.kernel_manager.max_loaded_time
                else None
            ),
            "instruments": [instrument.name for instrument in self.extraction_state.instruments],
            "satellite_name": self.extraction_state.instruments[0].satellite_name,
            "filter_name": self.extraction_state.filter.name,
            "extra_filters": {
                _instrument: _filter.name
                for _instrument, _filter in self.extraction_state.custom_filter_objects.items()
            },
            "frame": self.extraction_state.kernel_manager.main_reference_frame,
            "created_at": datetime.now(),
            "finished": False,
        }
        return Sessions.prepare_extraction_metadata(simulation_metadata)

    def iteration_housekeeping(self):
        for instr_state in self.instrument_states.values():
            if len(instr_state.reprojected_data) >= MONGO_UPLOAD_BATCH_SIZE:
                # If we have more than MONGO_UPLOAD_BATCH_SIZE, we can push to MongoDB
                import pdb

                pdb.set_trace()
                thread = Sessions.start_background_batch_insert(
                    instr_state.reprojected_data, instr_state.data_collection
                )
                instr_state.reprojected_data = []
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
                    data_entry,
                    self.instrument_states[instrument_name].instrument,
                    self.extraction_state.get_filter(instrument_name),
                )

                if reprojected_data:
                    # Add for traceability
                    reprojected_data["meta"] = {"extraction_name": self.extraction_state._extraction_name, "et": reprojected_data["et"]}
                    reprojected_data["timestamp"] = et2astropy_time(reprojected_data["et"]).datetime
                    del reprojected_data["et"]

                    self.instrument_states[instrument_name].reprojected_data.append(reprojected_data)
                    self.instrument_states[instrument_name].total_reprojected_data += 1
            except NotFoundError as e:
                # This is to be expected
                self.flush_SPICE()
            except Exception as e:
                SPICELog.log_spice_exception(e, context=f"Error processing data entry for instrument {instrument_name}")
                self.flush_SPICE()

    def setup_data_connectors(self) -> Dict[str, BaseDataConnector]:
        connectors = {}
        for name, interval_list in self.extraction_state.interval_manager.intervals.items():
            connectors[name] = DATA_CONNECTOR_MAP[name](interval_list, self.extraction_state.kernel_manager)
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
        self.instrument_states = {
            instr.name: self.InstrumentState(instr, self.data_connectors[instr.name])
            for instr in self.extraction_state.instruments
        }

        # Do the mongo DB
        pbar = (
            tqdm(
                total=len(self.extraction_state.interval_manager),
                disable=SUPRESS_TQDM,
                ncols=TQDM_NCOLS,
                desc="Processing Time Intervals",
            )
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
            instr.name: Sessions.prepare_extraction_collections(
                instr.name, self.data_connectors[instr.name].timeseries, self.data_connectors[instr.name].indices
            )
            for instr in self.extraction_state.instruments
        }

        # Potentially remove dangling old data from non-sucesfull task runs
        for instr in self.extraction_state.instruments:
            interval_list = interval_manager.get_interval_by_instrument(instr.name)
            Sessions.remove_potentail_data_from_failed_task_runs(
                interval_list.start_et,
                interval_list.end_et,
                self.extraction_state._extraction_name,
                self.data_collections[instr.name],
            )

        # Iterate through the Interval Queu
        with pbar:
            while True:
                try:
                    instrument_interval_tuple = self.extraction_state.interval_manager.next_interval()
                    if instrument_interval_tuple is None:
                        break

                    instrument_name, interval = instrument_interval_tuple
                    self.last_interval_start_et = interval.start_et

                    if interactive_progress:
                        pbar.set_description(f"Processing interval {interval} for instrument {instrument_name}")
                    else:
                        logger.info(f"Processing interval {interval} for instrument {instrument_name}")

                    new_data = self.data_connectors[instrument_name].read_interval(interval)
                    if new_data:
                        logger.info(
                            f"Instrument {instrument_name} has {len(new_data)} data points in interval {interval}"
                        )
                        self.instrument_states[instrument_name].data.extend(new_data)
                        self.instrument_states[instrument_name].total_data += len(new_data)
                    else:
                        logger.info(f"Instrument {instrument_name} has no data in interval {interval}")

                    self.process_data()
                except Exception as e:
                    SPICELog.log_spice_exception(
                        e, context=f"Error processing interval {interval} for instrument {instrument_name}"
                    )
                    self.flush_SPICE()
                finally:
                    import pdb; pdb.set_trace()
                    self.iteration_housekeeping()
                    try:
                        if not instrument_interval_tuple is None:
                            pbar.update(1)
                    except Exception as e:
                        # This is not to be expected, but does not really matter
                        pass

        # Set to infinity so all the remaining data would be processed
        self.last_interval_start_et = float("inf")
        self.process_data()

        self.iteration_housekeeping()

        # Await all threads.
        for thread in self.threads:
            if thread is not None:
                thread.join()

        Sessions.process_failed_inserts()

        update_thread = Sessions.start_background_update_extraction_metadata(
            self.extraction_state.simulation_metadata_id,
            self.extraction_state.end_time.datetime,
            finished=True,
            metadata={instr_name: instr_state.total_data for instr_name, instr_state in self.instrument_states.items()},
        )
        self.threads.append(update_thread)

        if current_task is not None:
            # Update the task state to SUCCESS
            current_task.update_state(state="SUCCESS", meta={"result": "Task processed sucesfully."})

        for thread in self.threads:
            if thread is not None:
                thread.join()
