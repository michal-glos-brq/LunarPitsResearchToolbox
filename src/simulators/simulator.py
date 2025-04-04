import time
import logging
from contextlib import nullcontext
from typing import Optional, List, Dict
from datetime import timedelta
from dataclasses import dataclass

import numpy as np
from celery import Task
from tqdm import tqdm
import spiceypy as spice
from astropy.time import Time, TimeDelta
from threading import Thread

from src.SPICE.kernel_utils.kernel_management import BaseKernelManager
from src.SPICE.instruments.instrument import BaseInstrument, ProjectionPoint
from src.db.interface import Sessions
from src.simulators.filters import BaseFilter
from src.simulators.config import (
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
from src.global_config import SUPRESS_TQDM


def log_spice_exception(e: Exception, context: str = ""):
    """
    Log the exception with its type and any SPICE error state.
    """
    err_type = type(e).__name__
    msg = f"{context} Exception: [{err_type}] {e}"
    if spice.failed():
        spice_error_message = spice.getmsg("SHORT")
        msg += f" | SPICE error: {spice_error_message}"
        spice.reset()
    logging.warning(msg)


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
                log_spice_exception(e, f"Initial height calculation for {instrument.name}")
                self.heights.add(instrument._height)
            try:
                fov_width = instrument.recalculate_bounds_to_boresight_distance(current_et)
                self.fov_widths.add(fov_width)
            except Exception as e:
                log_spice_exception(e, f"Initial FOV calculation for {instrument.name}")
                self.fov_widths.add(instrument._fov_width)

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
        def __init__(self, kernel_manager: BaseKernelManager):
            self.kernel_manager = kernel_manager
            self.simulation_timekeeper = timedelta(seconds=0)
            self.max_speed = DynamicMaxBuffer(DYNAMIC_MAX_BUFFER_SPACECRAFT_VELOCITY_SIZE)
            self.max_speed_counter = 0

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

    def check_threads(self):
        # Iterate in reverse to safely pop finished threads.
        for thread_id in range(len(self.threads) - 1, -1, -1):
            if self.threads[thread_id] is None:
                self.threads.pop(thread_id)

            elif not self.threads[thread_id].is_alive():
                self.threads[thread_id].join()

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
        distances_to_thresholds = []
        for instrument in self.instruments:
            instrument_state = self.instrument_simulation_states[instrument.name]
            distance = rank - (
                instrument_state.heights.maximum + instrument_state.fov_widths.maximum + self.filter.hard_radius
            )
            if distance > 0:
                distances_to_thresholds.append(distance)
                continue

            try:
                projection: ProjectionPoint = instrument.project_boresight(
                    self.simulation_state.current_simulation_timestamp_et
                )
                rank = self.filter.rank_point(projection.projection)
                distance = rank - (self.filter.hard_radius + instrument_state.fov_widths.maximum) * TOLERANCE_MARGIN
                if distance <= 0:
                    # Use simulation state's current timestamp.
                    datetime_current_timestamp = self.simulation_state.current_simulation_timestamp.to_datetime()
                    instrument_state.positive_sensing_batch.append(
                        {
                            "et": self.simulation_state.current_simulation_timestamp_et,
                            "astropy_utc": self.simulation_state.current_simulation_timestamp.utc.iso,
                            "timestamp_utc": datetime_current_timestamp,
                            "distance": distance,
                            "boresight": projection.projection.tolist(),
                            "meta": {
                                "min_loaded_time": self.kernel_manager.min_loaded_time.utc.iso,
                                "max_loaded_time": self.kernel_manager.max_loaded_time.utc.iso,
                                "simulation_start": self.start_time.utc.iso,
                                "satellite_position": spacecraft_position.tolist(),
                                "filter_name": self.filter.name,
                                "fov_width": instrument_state.fov_widths.maximum,
                                "height": instrument_state.heights.maximum,
                            },
                        }
                    )
                    instrument_state.total_success += 1
                else:
                    distances_to_thresholds.append(distance)
            except Exception as e:
                log_spice_exception(e, f"Error calculating projection for {instrument.name}")
                datetime_current_timestamp = self.simulation_state.current_simulation_timestamp.to_datetime()
                instrument_state.failed_computation_batch.append(
                    {
                        "et": self.simulation_state.current_simulation_timestamp_et,
                        "timestamp_utc": datetime_current_timestamp,
                        "astropy_utc": self.simulation_state.current_simulation_timestamp.utc.iso,
                        "error": str(e),
                        "meta": {
                            "min_loaded_time": self.kernel_manager.min_loaded_time.utc.iso,
                            "max_loaded_time": self.kernel_manager.max_loaded_time.utc.iso,
                            "simulation_start": self.start_time.utc.iso,
                            "satellite_position": spacecraft_position.tolist(),
                            "filter_name": self.filter.name,
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
                    log_spice_exception(e, f"Updating height for {instrument.name}")
                instrument_state.heights_counter = DYNAMIC_MAX_BUFFER_HEIGHT_UPDATE_RATE
            if instrument_state.fov_widths_counter == 0:
                try:
                    instrument_state.fov_widths.add(
                        instrument.recalculate_bounds_to_boresight_distance(
                            self.simulation_state.current_simulation_timestamp_et
                        )
                    )
                except Exception as e:
                    log_spice_exception(e, f"Updating FOV for {instrument.name}")
                instrument_state.fov_widths_counter = DYNAMIC_MAX_BUFFER_FOV_WIDTH_UPDATE_RATE

    def _simulation_step_housekeeping(
        self, pbar, interactive_progress: bool = True, current_task: Optional[Task] = None
    ):
        # Advance simulation time.
        self.simulation_state._time_step(self.computation_timedelta)
        # Periodically dump the state of simulation
        if time.time() - self.simulation_state.real_time > SIM_STATE_DUMP_INTERVAL:
            if interactive_progress:
                state_string_prefix = "========== SIMULATION STATE REPORT ==========\n"
                block_lines = [state.dump_status_lines() for state in self.instrument_simulation_states.values()]
                transposed = zip(*block_lines)
                aligned_output = "\n".join("   ".join(f"{s:<16}" for s in line_parts) for line_parts in transposed)
                state_string = (
                    aligned_output
                    + f"\n🚀  Locally max spacecraft speed: {self.simulation_state.max_speed.maximum:.2f} km/s\n"
                )
                state_string = state_string_prefix + state_string
                tqdm.write(state_string)
                if current_task is not None:
                    state = {
                        instrument.name: instrument_state.to_summary_dict()
                        for instrument, instrument_state in self.instrument_simulation_states.items()
                    }
                    state["sc_speed"] = self.simulation_state.max_speed.maximum
                    current_task.update_state(
                        state="PROGRESS",
                        meta={
                            "state": state,
                            "progress [%]": 100
                            * self.simulation_state.simulation_timekeeper.total_seconds()
                            / self.simulation_duration.total_seconds(),
                        },
                    )
            self.simulation_state.real_time = time.time()

        if interactive_progress:
            pbar.update(self.computation_timedelta.sec)
        else:
            logging.info(
                f"Simulated time: {RemoteSensingSimulator.format_td(self.simulation_state.simulation_timekeeper)} / {self.simulation_duration_formatted}"
            )

        # Flush data to the database.
        for instrument in self.instruments:
            instrument_state = self.instrument_simulation_states[instrument.name]
            if len(instrument_state.positive_sensing_batch) >= MONGO_PUSH_BATCH_SIZE:
                thread = Sessions.start_background_batch_insert(
                    instrument_state.positive_sensing_batch, instrument_state.success_collections
                )
                self.threads.append(thread)
                instrument_state.positive_sensing_batch = []
                self.check_threads()
            if len(instrument_state.failed_computation_batch) >= MONGO_PUSH_BATCH_SIZE:
                thread = Sessions.start_background_batch_insert(
                    instrument_state.failed_computation_batch, instrument_state.failed_collections
                )
                self.threads.append(thread)
                instrument_state.failed_computation_batch = []
                self.check_threads()

    def start_simulation(
        self,
        start_time: Optional[Time] = None,
        end_time: Optional[Time] = None,
        interactive_progress: bool = True,
        current_task: Optional[Task] = None,
    ):
        """
        :param start_time: Start time of the simulation
        :param end_time: End time of the simulation
        :param interactive_progress: Whether to show the progress bar, if false, just dump progress and logs
        :param current_task: Celery task object, if provided, will be used to update the progress
        """
        # Use kernel manager's loaded times if not provided.
        self.start_time = self.kernel_manager.min_loaded_time if start_time is None else start_time
        self.end_time = self.kernel_manager.max_loaded_time if end_time is None else end_time
        self.simulation_duration = timedelta(seconds=(self.end_time - self.start_time).sec)
        self.simulation_duration_formatted = RemoteSensingSimulator.format_td(self.simulation_duration)
        self.total_seconds = self.simulation_duration.total_seconds()

        # Initialize simulation state.
        self.simulation_state = self.SimulationState(self.kernel_manager)
        self.simulation_state._setup_time(self.start_time)

        # Initialize instrument simulation states using current ET.
        self.instrument_simulation_states = {
            instrument.name: self.InstrumentSimulationState(
                instrument, self.simulation_state.current_simulation_timestamp_et
            )
            for instrument in self.instruments
        }

        pbar = tqdm(total=self.total_seconds, unit="s", disable=SUPRESS_TQDM) if interactive_progress else nullcontext()

        with pbar:
            while self.simulation_state.current_simulation_timestamp < self.end_time:
                try:
                    self._simulation_step()
                except Exception as e:
                    log_spice_exception(e, "Error during simulation step")
                finally:
                    self._simulation_step_housekeeping(pbar, interactive_progress, current_task)

        # Final flush to the database.
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
