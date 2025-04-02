import time
import logging
from contextlib import nullcontext
from typing import Optional, List, Dict
from dataclasses import dataclass

import numpy as np
from celery import Task
from tqdm import tqdm
import spiceypy as spice
from datetime import timedelta
from astropy.time import Time, TimeDelta
from pymongo.collection import Collection

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


# from src.config import LRO_SPEED, TIME_STEP, MAX_TIME_STEP, SIMULATION_BATCH_SIZE


class DynamicMaxBuffer:
    """
    This class is used to create a dynamic buffered max value.
    """

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size))
        self.index = 0
        self.maximum = float("-inf")
        self.maximum_ttl = buffer_size

    def add(self, value: float):
        self.buffer[self.index] = value

        if value >= self.maximum:
            # New max found — reset TTL
            self.maximum = value
            self.maximum_ttl = self.buffer_size
        else:
            # Not the new max — decrease TTL
            self.maximum_ttl -= 1
            if self.maximum_ttl == 0:
                # TTL expired — recompute max + reset TTL
                self.maximum = self.buffer.max()
                # Find index of current max value to estimate new TTL
                max_pos = np.argmax(self.buffer)
                distance = (max_pos - self.index) % self.buffer_size
                self.maximum_ttl = distance if distance != 0 else self.buffer_size

        self.index = (self.index + 1) % self.buffer_size


@dataclass
class InstrumentSimulationState:
    success_collections: Collection
    failed_collections: Collection
    positive_sensing_batch: List[Dict]
    failed_computation_batch: List[Dict]
    heights: DynamicMaxBuffer
    fov_widths: DynamicMaxBuffer
    heights_counter: int = DYNAMIC_MAX_BUFFER_HEIGHT_SIZE
    fov_widths_counter: int = DYNAMIC_MAX_BUFFER_FOV_WIDTH_SIZE
    total_success: int = 0
    total_failed: int = 0
    
    def dump_status_lines(self) -> List[str]:
        name = self.success_collections.name[:10]  # Shorten if too long
        return [
            f"{name:^12}",  # Centered
            f"✅ {len(self.positive_sensing_batch):>4} | T {self.total_success:<5}",
            f"❌ {len(self.failed_computation_batch):>4} | T {self.total_failed:<5}",
            f"📏 {self.heights.maximum:>8.2f}",
            f"🎯 {self.fov_widths.maximum:>8.2f}",
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


class RemoteSensingSimulator:
    """
    This class is used to simulate the trajectory and orientation of spacecraft and identify time
    intervals when it was pointing towards some area of our interest on the lunar surface
    
    Simulation is computationally optimized for a single satellite, but list of instruments.
    KernelManager has to be pre-configured, only step method is called within the simulation
    """

    
    def __init__(self, instruments: List[BaseInstrument], filter_object: BaseFilter, kernel_manager: BaseKernelManager):
        self.instruments = instruments
        self.filter = filter_object
        self.kernel_manager = kernel_manager
        self._computation_timedelta = SIMULATION_STEP
        assert len(set([instrument.satellite_name for instrument in self.instruments])) == 1, f"All instruments have to be from the same satellite, but got {set([instrument.satellite_name for instrument in self.instruments])}"

    @property
    def computation_timedelta(self):
        return TimeDelta(self._computation_timedelta, format="sec")

    def _setup_time(self, time: Time):
        """Setup or reset the simulation timing"""
        # initialize the simulation state
        self.current_simulation_step = 0
        self.current_simulation_timestamp_et = spice.str2et(time.utc.iso)
        self.current_simulation_timestamp = time
        self.kernel_manager.step(time)


    def _simulation_step(self):
        """Perform a single step the simulation itself"""
        self.current_simulation_timestamp += self.computation_timedelta
        self.current_simulation_timestamp_et = spice.str2et(self.current_simulation_timestamp.utc.iso)
        self.current_simulation_step += 1
        self.kernel_manager.step(self.current_simulation_timestamp)


    def start_simulation(self, start_time: Optional[Time] = None, end_time: Optional[Time] = None, interactive_progress: bool = True, current_task: Optional[Task] = None):
        """
        Start the simulation for the given time interval.
        :param start_time: Start time of the simulation
        :param end_time: End time of the simulation
        :param interactive_progress: Whether to show the progress bar, if false, just dump progress and logs
        :param current_task: Celery task object, if provided, will be used to update the progress
        """

        start_time = self.kernel_manager.min_loaded_time if start_time is None else start_time
        end_time = self.kernel_manager.max_loaded_time if end_time is None else end_time

        self._setup_time(start_time)
        real_time = time.time()

        self.instrument_simulation_states = {}
        for instrument in self.instruments:
            success_collection, failed_collection = Sessions.prepare_simulation_collections(instrument.name)
            instrument_state = InstrumentSimulationState(
                success_collections=success_collection,
                failed_collections=failed_collection,
                positive_sensing_batch=[],
                failed_computation_batch=[],
                heights=DynamicMaxBuffer(DYNAMIC_MAX_BUFFER_HEIGHT_SIZE),
                fov_widths=DynamicMaxBuffer(DYNAMIC_MAX_BUFFER_FOV_WIDTH_SIZE)
            )
            try:
                instrument_state.heights.add(np.linalg.norm(instrument.project_boresight(self.current_simulation_timestamp_et).spacecraft_relative))
            except Exception as e:
                instrument_state.heights.add(instrument._height)
            try:
                instrument_state.fov_widths.add(instrument.recalculate_bounds_to_boresight_distance(self.current_simulation_timestamp_et))
            except Exception as e:
                instrument_state.fov_widths.add(instrument._fov_width)
            self.instrument_simulation_states[instrument.name] = instrument_state

        max_speed = DynamicMaxBuffer(DYNAMIC_MAX_BUFFER_SPACECRAFT_VELOCITY_SIZE)
        max_speed_counter = 0
        threads = []

        def check_threads():
            for thread_id in range(len(threads)-1, -1, -1):
                if not threads[thread_id].is_alive():
                    threads[thread_id].join()
                    threads.pop(thread_id)

        def format_td(td: timedelta) -> str:
            total = int(td.total_seconds())
            return f"{total // 3600:02}:{(total % 3600) // 60:02}:{total % 60:02}"

        total_seconds = (end_time - start_time).sec
        simulation_duration = timedelta(seconds=(end_time - start_time).sec)
        simulation_timekeeper, simulation_duration_formatted = timedelta(seconds=0), format_td(simulation_duration)
        pbar = tqdm(total=total_seconds, unit="time_step") if interactive_progress else nullcontext()

        with pbar:
            while self.current_simulation_timestamp < end_time:
                try:

                    # Early rejection based on spacecraft position
                    if max_speed_counter == 0:
                        max_speed_counter = DYNAMIC_MAX_BUFFER_SPACECRAFT_VELOCITY_UPDATE_RATE
                        spacecraft_position, spacecraft_velocity = self.instruments[0].calculate_spacecraft_position_and_velocity(self.current_simulation_timestamp_et, self.kernel_manager.main_reference_frame)
                        max_speed.add(np.linalg.norm(spacecraft_velocity))
                    else:
                        spacecraft_position = self.instruments[0].calculate_spacecraft_position(self.current_simulation_timestamp_et, self.kernel_manager.main_reference_frame)
                        max_speed_counter -= 1

                    rank = self.filter.rank_point(spacecraft_position)

                    distances_to_tresholds = []
                    for instrument in self.instruments:

                        instrument_state: InstrumentSimulationState = self.instrument_simulation_states[instrument.name]

                        # Check for fast dismissal with spacecraft position
                        if (distance := (rank - (instrument_state.heights.maximum + instrument_state.fov_widths.maximum + self.filter.hard_radius))) > 0:
                            distances_to_tresholds.append(distance)
                            continue

                        try:
                            # Went through screening, now actually compute the state
                            projection: ProjectionPoint = instrument.project_boresight(self.current_simulation_timestamp_et)
                            # Check how close is the closest point    
                            rank = self.filter.rank_point(projection.projection)

                            # Check if the projection is within the FOV
                            if (distance := rank - (self.filter.hard_radius + instrument_state.fov_widths.maximum) * TOLERANCE_MARGIN) <= 0:
                                # Add to the batch
                                datetime_current_timestamp = self.current_simulation_timestamp.to_datetime()
                                instrument_state.positive_sensing_batch.append({
                                    "et": self.current_simulation_timestamp_et,
                                    "timestamp_utc": datetime_current_timestamp,
                                    "distance": distance,
                                    "boresight": projection.projection.tolist(),
                                    "meta": {
                                        "astropy_offset": (self.current_simulation_timestamp - Time(datetime_current_timestamp, scale="utc")).sec,
                                        "satellite_position": spacecraft_position.tolist(),
                                        "filter_name": self.filter.name,
                                        "fov_width": instrument_state.fov_widths.maximum,
                                        "height": instrument_state.heights.maximum,
                                    }
                                })
                                instrument_state.total_success += 1
                            # If projection is outside of the FOV, we can prolong the step to skip large lunar swaths of no interest
                            # With computed satellite speed and distance needed to reach the closest treshold (we use lower bound, so just fov width and hard treshold is used)
                            else:
                                distances_to_tresholds.append(distance)

                        except Exception as e:
                            # We have an error, possibly global in spice, so let's not sabotage the whole sim.
                            logging.warning(f"Error calculating projection ({instrument.name}): {e}")
                            datetime_current_timestamp = self.current_simulation_timestamp.to_datetime()
                            instrument_state.failed_computation_batch.append({
                                "et": self.current_simulation_timestamp_et,
                                "timestamp_utc": datetime_current_timestamp,
                                "error": str(e),
                                "meta": {
                                    "astropy_offset": (self.current_simulation_timestamp - Time(datetime_current_timestamp, scale="utc")).sec,
                                    "satellite_position": spacecraft_position.tolist(),
                                    "filter_name": self.filter.name,
                                }
                            })
                            instrument_state.total_failed += 1
                            continue


                        # In case it's required by "timer", recalculate the FOV size and height
                        instrument_state.heights_counter, instrument_state.fov_widths_counter = instrument_state.heights_counter - 1, instrument_state.fov_widths_counter - 1
                        if instrument_state.heights_counter == 0:
                            instrument_state.heights.add(np.linalg.norm(projection.spacecraft_relative))
                            instrument_state.heights_counter = DYNAMIC_MAX_BUFFER_HEIGHT_UPDATE_RATE
                        if instrument_state.fov_widths_counter == 0:
                            instrument_state.fov_widths.add(instrument.recalculate_bounds_to_boresight_distance(self.current_simulation_timestamp_et))
                            instrument_state.fov_widths_counter = DYNAMIC_MAX_BUFFER_FOV_WIDTH_UPDATE_RATE

                    # If we have a distance to tresholds, we can use it to prolong the step
                    if len(distances_to_tresholds) > 0:
                        self._computation_timedelta = max(SIMULATION_STEP, min(distances_to_tresholds) / max_speed.maximum)
                    else:
                        self._computation_timedelta = SIMULATION_STEP

                except Exception as e:
                    logging.warning(f"Error during simulation step: {e}")


                finally:
                    # Simulation maintnance, have to be always executed!
                    self._simulation_step()
                    simulation_timekeeper += timedelta(seconds=self.computation_timedelta.sec)

                    # Periodically dump the state of simulation
                    if time.time() - real_time > SIM_STATE_DUMP_INTERVAL:
                        if interactive_progress:
                            state_string_prefix = "========== SIMULATION STATE REPORT ==========\n"
                    
                            block_lines = [state.dump_status_lines() for state in self.instrument_simulation_states.values()]
                            transposed = zip(*block_lines)
                    
                            aligned_output = "\n".join("   ".join(f"{s:<16}" for s in line_parts) for line_parts in transposed)
                            state_string = aligned_output + f"\n🚀  Locally max spacecraft speed: {max_speed.maximum:.2f} km/s\n"
                            state_string = state_string_prefix + state_string

                            tqdm.write(state_string)
                            if current_task is not None:
                                state = {instrument.name: instrument_state.to_summary_dict() for instrument, instrument_state in self.instrument_simulation_states.items()}
                                state["sc_speed"] = max_speed.maximum
                                current_task.update_state(state="PROGRESS", meta={"state": state, "progress [%]": 100 * simulation_timekeeper / simulation_duration})

                        real_time = time.time()


                    if interactive_progress:
                        pbar.update(self.computation_timedelta.sec)
                    else:
                        logging.info(f"Simulated time: {format_td(simulation_timekeeper)} / {simulation_duration_formatted}")


                    # Push data to the database
                    for instrument in self.instruments:

                        instrument_state: InstrumentSimulationState = self.instrument_simulation_states[instrument.name]
                        if len(instrument_state.positive_sensing_batch) >= MONGO_PUSH_BATCH_SIZE:
                            thread = Sessions.start_background_batch_insert(instrument_state.positive_sensing_batch, instrument_state.success_collections)
                            threads.append(thread)
                            instrument_state.positive_sensing_batch = []
                            check_threads()
            
                        if len(instrument_state.failed_computation_batch) >= MONGO_PUSH_BATCH_SIZE:
                            thread = Sessions.start_background_batch_insert(instrument_state.failed_computation_batch, instrument_state.failed_collections)
                            threads.append(thread)
                            instrument_state.failed_computation_batch = []
                            check_threads()

        # Final push to the database
        for instrument in self.instruments:
            instrument_state: InstrumentSimulationState = self.instrument_simulation_states[instrument.name]
            if len(instrument_state.positive_sensing_batch) > 0:
                thread = Sessions.start_background_batch_insert(instrument_state.positive_sensing_batch, instrument_state.success_collections)
                threads.append(thread)

            if len(instrument_state.failed_computation_batch) > 0:
                thread = Sessions.start_background_batch_insert(instrument_state.failed_computation_batch, instrument_state.failed_collections)
                threads.append(thread)

        # Since we want to kill all threads when the main thread exits, let's await them just to be sure!
        for thread in threads:
            thread.join()
