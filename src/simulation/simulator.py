import time
import logging
import datetime
from contextlib import nullcontext
from typing import Optional, List, Dict
from datetime import timedelta
from bson import ObjectId

import numpy as np
from celery import Task
from tqdm import tqdm
import spiceypy as spice
from astropy.time import Time, TimeDelta
from threading import Thread

from src.SPICE.utils import SPICELog
from src.SPICE.kernel_utils.kernel_management import BaseKernelManager
from src.SPICE.instruments.instrument import BaseInstrument, ProjectionPoint
from src.db.interface import Sessions
from src.SPICE.filters import BaseFilter
from src.simulation.config import (
    SIMULATION_STEP,
    DYNAMIC_MAX_BUFFER_FOV_WIDTH_SIZE,
    DYNAMIC_MAX_BUFFER_FOV_WIDTH_UPDATE_RATE,
    DYNAMIC_MAX_BUFFER_HEIGHT_SIZE,
    DYNAMIC_MAX_BUFFER_HEIGHT_UPDATE_RATE,
    MONGO_PUSH_BATCH_SIZE,
    TOLERANCE_MARGIN,
    DYNAMIC_MAX_BUFFER_SPACECRAFT_VELOCITY_SIZE,
    DYNAMIC_MAX_BUFFER_SPACECRAFT_VELOCITY_UPDATE_RATE,
    SIM_STATE_DUMP_INTERVAL,
)
from src.global_config import SUPRESS_TQDM, TQDM_NCOLS



class DynamicMaxBuffer:
    """
    This class creates a dynamic, buffered maximum value.
    """

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = np.zeros(buffer_size)
        self.index = 0
        self.maximum = float("-inf")
        self.maximum_ttl = buffer_size

    def add(self, value: float):
        self.buffer[self.index] = value
        if value >= self.maximum:
            # New max found – reset TTL
            self.maximum = value
            self.maximum_ttl = self.buffer_size
        else:
            # Not the new max – decrease TTL
            self.maximum_ttl -= 1
            if self.maximum_ttl == 0:
                # TTL expired – recompute max and reset TTL
                self.maximum = self.buffer.max()
                max_pos = np.argmax(self.buffer)
                distance = (max_pos - self.index) % self.buffer_size
                self.maximum_ttl = distance if distance != 0 else self.buffer_size
        self.index = (self.index + 1) % self.buffer_size


class RemoteSensingSimulator:
    """
    Simulate the trajectory and orientation of a spacecraft to identify time intervals
    when it points toward an area of interest on the lunar surface.

    Optimized for a single satellite with multiple instruments.
    The KernelManager must be pre-configured; only the step method is called within the simulation.
    """

    class InstrumentSimulationState:
        def __init__(self, instrument: BaseInstrument, current_et: float):
            self.success_collections, self.failed_collections = Sessions.prepare_simulation_collections(instrument.name)
            self.positive_sensing_batch: List[Dict] = []
            self.failed_computation_batch: List[Dict] = []
            self.heights = DynamicMaxBuffer(DYNAMIC_MAX_BUFFER_HEIGHT_SIZE)
            self.fov_widths = DynamicMaxBuffer(DYNAMIC_MAX_BUFFER_FOV_WIDTH_SIZE)
            self.heights_counter: int = DYNAMIC_MAX_BUFFER_HEIGHT_SIZE
            self.fov_widths_counter: int = DYNAMIC_MAX_BUFFER_FOV_WIDTH_SIZE
            self.total_success: int = 0
            self.total_failed: int = 0
            try:
                height = np.linalg.norm(instrument.project_boresight(current_et).spacecraft_relative)
                self.heights.add(height)
            except Exception as e:
                self.heights.add(instrument._height)
                SPICELog.log_spice_exception(e, f"Initial height calculation for {instrument.name}")
            try:
                fov_width = instrument.recalculate_bounds_to_boresight_distance(current_et)
                self.fov_widths.add(fov_width)
            except Exception as e:
                self.fov_widths.add(instrument._fov_width)
                SPICELog.log_spice_exception(e, f"Initial FOV calculation for {instrument.name}")

        def dump_status_lines(self) -> List[str]:
            name = self.success_collections.name[:22]
            return [
                f"{name:^24}",
                f"✅ {len(self.positive_sensing_batch):>4} | T {self.total_success:<5}".ljust(24),
                f"❌ {len(self.failed_computation_batch):>4} | T {self.total_failed:<5}".ljust(24),
                f"📏 Max Height: {self.heights.maximum:>7.2f}".ljust(24),
                f"🎯 Max FOV:    {self.fov_widths.maximum:>7.2f}".ljust(24),
            ]

        def to_summary_dict(self) -> Dict[str, float]:
            return {
                "collection": self.success_collections.name,
                "positive_batch_size": len(self.positive_sensing_batch),
                "failed_batch_size": len(self.failed_computation_batch),
                "total_success": self.total_success,
                "total_failed": self.total_failed,
                "max_height": round(self.heights.maximum, 3),
                "max_fov_width": round(self.fov_widths.maximum, 3),
            }

    class SimulationState:
        def __init__(self, kernel_manager: BaseKernelManager, satellite_name: str):
            self.kernel_manager = kernel_manager
            self.simulation_timekeeper = timedelta(seconds=0)
            self.max_speed = DynamicMaxBuffer(DYNAMIC_MAX_BUFFER_SPACECRAFT_VELOCITY_SIZE)
            self.max_speed_counter = 0
            self.satellite_name = satellite_name
            self.spacecraft_position_computation_failed = []
            self.spacecraft_position_computation_failed_counter = 0
            self.spacecraft_position_computation_failed_collection = Sessions.get_spacecraft_position_failed_collection(
                satellite_name
            )

        def _setup_time(self, time_obj: Time):
            """Initialize the simulation timing. Can be used for reset too."""
            self.current_simulation_step = 0
            self.current_simulation_timestamp_et = spice.str2et(time_obj.utc.iso)
            self.current_simulation_timestamp = time_obj
            self.kernel_manager.step(time_obj)
            self.real_time = time.time()

        def _time_step(self, dt: TimeDelta):
            """Advance the simulation by dt."""
            self.current_simulation_timestamp += dt
            self.simulation_timekeeper += timedelta(seconds=dt.sec)
            self.current_simulation_timestamp_et = spice.str2et(self.current_simulation_timestamp.utc.iso)
            self.current_simulation_step += 1
            self.kernel_manager.step(self.current_simulation_timestamp)

    def __init__(self, instruments: List[BaseInstrument], filter_object: BaseFilter, kernel_manager: BaseKernelManager):
        self.instruments = instruments
        self.filter = filter_object
        self.kernel_manager = kernel_manager
        self._computation_timedelta = SIMULATION_STEP
        self.threads: List[Thread] = []
        # Ensure all instruments are from the same satellite.
        if len({instrument.satellite_name for instrument in self.instruments}) != 1:
            raise ValueError(
                f"All instruments must be from the same satellite, but got {[instrument.satellite_name for instrument in self.instruments]}"
            )

    @property
    def computation_timedelta(self) -> TimeDelta:
        return TimeDelta(self._computation_timedelta, format="sec")

    @staticmethod
    def format_td(td: timedelta) -> str:
        total = int(td.total_seconds())
        return f"{total // 3600:02}:{(total % 3600) // 60:02}:{total % 60:02}"

    @property
    def simulation_quality_metadata(self):
        return {
            "total_simulated_seconds": int(self.simulation_state.simulation_timekeeper.total_seconds()),
            "total_simulated_steps": self.simulation_state.current_simulation_step,
            "instruments_summary": {
                instrument: {
                    "total_success": state.total_success,
                    "total_failed": state.total_failed,
                }
                for instrument, state in self.instrument_simulation_states.items()
            },
            "spacecraft_position_computation_failed_total": self.simulation_state.spacecraft_position_computation_failed_counter,
        }


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
            spice.utc2et(self.simulation_state.current_simulation_timestamp.utc.iso)
        except Exception as e:
            ...

    def _simulation_step(self):
        # Early rejection based on spacecraft position.
        if self.simulation_state.max_speed_counter == 0:
            self.simulation_state.max_speed_counter = DYNAMIC_MAX_BUFFER_SPACECRAFT_VELOCITY_UPDATE_RATE
            spacecraft_position, spacecraft_velocity = self.instruments[0].calculate_spacecraft_position_and_velocity(
                self.simulation_state.current_simulation_timestamp_et, self.kernel_manager.main_reference_frame
            )
            self.simulation_state.max_speed.add(np.linalg.norm(spacecraft_velocity))
        else:
            spacecraft_position = self.instruments[0].calculate_spacecraft_position(
                self.simulation_state.current_simulation_timestamp_et, self.kernel_manager.main_reference_frame
            )
            self.simulation_state.max_speed_counter -= 1

        rank = self.filter.rank_point(spacecraft_position)

        for instrument in self.instruments:
            instrument_state = self.instrument_simulation_states[instrument.name]
            score = rank - (
                instrument_state.heights.maximum + instrument_state.fov_widths.maximum + self.filter.hard_radius
            )
            if score > 0:
                continue

            try:
                projection: ProjectionPoint = instrument.project_boresight(
                    self.simulation_state.current_simulation_timestamp_et
                )
                rank = self.filter.rank_point(projection.projection)
                score = rank - (self.filter.hard_radius + instrument_state.fov_widths.maximum) * TOLERANCE_MARGIN
                if score <= 0:
                    # Use simulation state's current timestamp.
                    datetime_current_timestamp = self.simulation_state.current_simulation_timestamp.to_datetime()
                    instrument_state.positive_sensing_batch.append(
                        {
                            "et": self.simulation_state.current_simulation_timestamp_et,
                            "timestamp_utc": datetime_current_timestamp,
                            "distance": rank,
                            "boresight": projection.projection.tolist(),
                            "meta": {
                                "score": score,
                                "simulation_id": self.simulation_metadata_id,
                                "satellite_position": spacecraft_position.tolist(),
                                "distance_margin": rank - self.filter.hard_radius,
                                "height": instrument_state.heights.maximum,
                            },
                        }
                    )
                    instrument_state.total_success += 1

            except Exception as e:
                SPICELog.log_spice_exception(e, f"Error calculating projection for {instrument.name}")
                self.flush_SPICE()

                datetime_current_timestamp = self.simulation_state.current_simulation_timestamp.to_datetime()
                instrument_state.failed_computation_batch.append(
                    {
                        "et": self.simulation_state.current_simulation_timestamp_et,
                        "timestamp_utc": datetime_current_timestamp,
                        "error": str(e),
                        "meta": {
                            "simulation_id": self.simulation_metadata_id,
                            "satellite_position": spacecraft_position.tolist(),
                        },
                    }
                )
                instrument_state.total_failed += 1
                continue

            # Update dynamic buffers.
            instrument_state.heights_counter -= 1
            instrument_state.fov_widths_counter -= 1
            if instrument_state.heights_counter == 0:
                try:
                    instrument_state.heights.add(np.linalg.norm(projection.spacecraft_relative))
                except Exception as e:
                    SPICELog.log_spice_exception(e, f"Updating height for {instrument.name}")
                instrument_state.heights_counter = DYNAMIC_MAX_BUFFER_HEIGHT_UPDATE_RATE
            if instrument_state.fov_widths_counter == 0:
                try:
                    instrument_state.fov_widths.add(
                        instrument.recalculate_bounds_to_boresight_distance(
                            self.simulation_state.current_simulation_timestamp_et
                        )
                    )
                except Exception as e:
                    SPICELog.log_spice_exception(e, f"Updating FOV for {instrument.name}")
                instrument_state.fov_widths_counter = DYNAMIC_MAX_BUFFER_FOV_WIDTH_UPDATE_RATE

    def simulation_step(self):
        """
        Perform a single simulation step.
        This method should be called in a loop to advance the simulation.
        """
        try:
            self._simulation_step()
        except Exception as e:
            self.simulation_state.spacecraft_position_computation_failed.append(
                {
                    "et": self.simulation_state.current_simulation_timestamp_et,
                    "timestamp_utc": self.simulation_state.current_simulation_timestamp.to_datetime(),
                    "error": str(e),
                    "satellite_name": self.simulation_state.satellite_name,
                }
            )
            self.simulation_state.spacecraft_position_computation_failed_counter += 1
            SPICELog.log_spice_exception(e, "Error during simulation step")
            self.flush_SPICE()

    def _simulation_step_housekeeping(
        self, pbar, interactive_progress: bool = True, current_task: Optional[Task] = None
    ):
        # Advance simulation time.
        self.simulation_state._time_step(self.computation_timedelta)
        # Periodically dump the state of simulation
        if time.time() - self.simulation_state.real_time > SIM_STATE_DUMP_INTERVAL:
            state_string_prefix = "========== SIMULATION STATE REPORT ==========\n"
            state_string_prefix += (
                f"Simulation name: {self.simulation_name}; simulation id: {self.simulation_metadata_id}\n"
            )
            block_lines = [state.dump_status_lines() for state in self.instrument_simulation_states.values()]
            transposed = zip(*block_lines)
            aligned_output = "\n".join("   ".join(f"{s:<16}" for s in line_parts) for line_parts in transposed)
            state_string = (
                aligned_output
                + f"\n🚀  Locally max spacecraft speed: {self.simulation_state.max_speed.maximum:.2f} km/s\n"
                + f"Failed computations: {self.simulation_state.spacecraft_position_computation_failed_counter}\n"
            )
            state_string = state_string_prefix + state_string

            if interactive_progress:
                tqdm.write(state_string)
            else:
                logging.info(state_string)
                logging.info(
                    f"Simulated time: {RemoteSensingSimulator.format_td(self.simulation_state.simulation_timekeeper)} / {self.simulation_duration_formatted}"
                )

            if current_task is not None:
                state = {
                    instrument: instrument_state.to_summary_dict()
                    for instrument, instrument_state in self.instrument_simulation_states.items()
                }
                state["sc_speed"] = self.simulation_state.max_speed.maximum
                current_task.update_state(
                    state="PROGRESS",
                    meta={
                        "timestamp": datetime.datetime.now(),
                        "state": state,
                        "progress [%]": 100
                        * self.simulation_state.simulation_timekeeper.total_seconds()
                        / self.simulation_duration.total_seconds(),
                    },
                )

            update_thread = Sessions.start_background_update_simulation_metadata(
                self.simulation_metadata_id,
                self.simulation_state.current_simulation_timestamp.utc.iso,
                metadata=self.simulation_quality_metadata,
            )
            self.threads.append(update_thread)
            self.simulation_state.real_time = time.time()

        if interactive_progress:
            pbar.update(self.computation_timedelta.sec)

        # Flush data to the database.
        if len(self.simulation_state.spacecraft_position_computation_failed) >= MONGO_PUSH_BATCH_SIZE:
            thread = Sessions.start_background_batch_insert(
                self.simulation_state.spacecraft_position_computation_failed,
                self.simulation_state.spacecraft_position_computation_failed_collection,
            )
            self.threads.append(thread)
            self.simulation_state.spacecraft_position_computation_failed = []

        for instrument in self.instruments:
            instrument_state = self.instrument_simulation_states[instrument.name]
            if len(instrument_state.positive_sensing_batch) >= MONGO_PUSH_BATCH_SIZE:
                thread = Sessions.start_background_batch_insert(
                    instrument_state.positive_sensing_batch, instrument_state.success_collections
                )
                self.threads.append(thread)
                instrument_state.positive_sensing_batch = []

            if len(instrument_state.failed_computation_batch) >= MONGO_PUSH_BATCH_SIZE:
                thread = Sessions.start_background_batch_insert(
                    instrument_state.failed_computation_batch, instrument_state.failed_collections
                )
                self.threads.append(thread)
                instrument_state.failed_computation_batch = []

            self.check_threads()

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
            "base_step": SIMULATION_STEP,
        }
        return Sessions.prepare_simulation_metadata(simulation_metadata)

    def start_simulation(
        self,
        start_time: Optional[Time] = None,
        end_time: Optional[Time] = None,
        interactive_progress: bool = True,
        supress_error_logs: bool = False,
        current_task: Optional[Task] = None,
        simulation_name: Optional[str] = None,
        task_group_id: Optional[str] = None,
        retry_count: Optional[int] = None,
    ):
        """
        :param start_time: Start time of the simulation
        :param end_time: End time of the simulation
        :param interactive_progress: Whether to show the progress bar, if false, just dump progress and logs
        :param current_task: Celery task object, if provided, will be used to update the progress
        :param supress_error_logs: Whether to suppress error logs
        :param simulation_name: Name of the simulation
        :param task_group_id: Task group ID for the simulation
        :param retry_count: Is that a retry of a task which already ran? Which retry?
        """
        # Use kernel manager's loaded times if not provided.
        self.start_time = self.kernel_manager.min_loaded_time if start_time is None else start_time
        self.end_time = self.kernel_manager.max_loaded_time if end_time is None else end_time
        self.simulation_duration = timedelta(seconds=(self.end_time - self.start_time).sec)
        self.simulation_duration_formatted = RemoteSensingSimulator.format_td(self.simulation_duration)
        self.total_seconds = self.simulation_duration.total_seconds()
        self._simulation_name = simulation_name
        self.simulation_name = f"{simulation_name}_{retry_count}" if retry_count is not None else simulation_name
        self.task_group_id = task_group_id

        # Initialize simulation state.
        self.simulation_state = self.SimulationState(self.kernel_manager, self.instruments[0].satellite_name)
        self.simulation_state._setup_time(self.start_time)

        # Initialize instrument simulation states using current ET.
        self.instrument_simulation_states = {
            instrument.name: self.InstrumentSimulationState(
                instrument, self.simulation_state.current_simulation_timestamp_et
            )
            for instrument in self.instruments
        }

        pbar = (
            tqdm(total=self.total_seconds, unit="s", disable=SUPRESS_TQDM, ncols=TQDM_NCOLS)
            if interactive_progress
            else nullcontext()
        )

        # Log the simulation state into Mongo
        task_already_finished = self.create_metadata_record()
        if task_already_finished:
            if current_task is not None:
                current_task.update_state(state="SUCCESS", meta={"result": "Task already finished, no computation to do."})
            return

        SPICELog.interactive_progress = interactive_progress
        SPICELog.supress_output = supress_error_logs

        with pbar:
            while self.simulation_state.current_simulation_timestamp < self.end_time:
                try:
                    self.simulation_step()
                except Exception as e:
                    SPICELog.log_spice_exception(e, "Error during simulation step")
                finally:
                    self._simulation_step_housekeeping(pbar, interactive_progress, current_task)

        # Final flush to the database.
        if self.simulation_state.spacecraft_position_computation_failed:
            Sessions.start_background_batch_insert(
                self.simulation_state.spacecraft_position_computation_failed,
                self.simulation_state.spacecraft_position_computation_failed_collection,
            )

        for instrument in self.instruments:
            instrument_state = self.instrument_simulation_states[instrument.name]
            if instrument_state.positive_sensing_batch:
                thread = Sessions.start_background_batch_insert(
                    instrument_state.positive_sensing_batch, instrument_state.success_collections
                )
                self.threads.append(thread)
            if instrument_state.failed_computation_batch:
                thread = Sessions.start_background_batch_insert(
                    instrument_state.failed_computation_batch, instrument_state.failed_collections
                )
                self.threads.append(thread)

        # Await all threads.
        for thread in self.threads:
            thread.join()

        Sessions.process_failed_inserts()

        update_thread = Sessions.start_background_update_simulation_metadata(
            self.simulation_metadata_id,
            self.simulation_state.current_simulation_timestamp.utc.iso,
            finished=True,
            metadata=self.simulation_quality_metadata,
        )
        self.threads.append(update_thread)
