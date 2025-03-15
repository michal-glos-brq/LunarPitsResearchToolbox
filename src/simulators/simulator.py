from typing import Optional

from tqdm import tqdm
import spiceypy as spice
from astropy.time import Time, TimeDelta

from src.global_config import TQDM_NCOLS
from src.kernel_utils.kernel_management import LunarKernelManager
from src.instruments.instrument import BaseInstrument
from src.db.mongo.interface import Sessions
from src.simulators.filters import BaseFilter
from src.config import LRO_SPEED, TIME_STEP, MAX_TIME_STEP, SIMULATION_BATCH_SIZE

class RemoteSensingSimulator:
    """This class is used to simulate the trajectory and orientation of spacecraft and identify time
    intervals when it was pointing towards some area of our interest on the lunar surface"""

    
    def __init__(self, instrument: BaseInstrument, filter: BaseFilter, kernel_manager: LunarKernelManager):
        self.instrument = instrument
        self.filter = filter
        self.kernel_manager = kernel_manager

    def _setup_time(self, time: Optional[Time] = None):
        """Setup or reset the simulation timing"""
        time = self.kernel_manager.min_loaded_time if time is None else time
        self.current_simulation_step = 0
        self.current_simulation_timestamp_et = spice.str2et(time.utc.iso)
        self.current_simulation_timestamp = time
        self.kernel_manager.step(time)

    def _simulation_step(self):
        """Step the simulation itself"""
        self.current_simulation_timestamp += self.computation_timedelta
        self.current_simulation_timestamp_et = spice.str2et(self.current_simulation_timestamp.utc.iso)
        self.current_simulation_step += 1
        self.sweep_iterator.step(self.current_simulation_timestamp)


    def _adjust_timestep(self, min_distance: float):
        new_time_step = (min_distance - self.rough_treshold) / LRO_SPEED
        self.computation_timedelta = TimeDelta(min(max(TIME_STEP, new_time_step), MAX_TIME_STEP), format="sec")
        self.adjusted_timesteps.append(new_time_step)
        self.min_distances.append(min_distance)


    def _calculate_state(self):
        '''Calculate the projection of current instrument boresight'''
        try:
            # TODO: Instrument the persistant simulation state
            # TODO: Maybe eventually group by satellite
            # Projects to the lunar surface and looks for closest points (may be empty)
            intersection = self.instrument.project_mean_boresight(self.current_simulation_timestamp_et)
            distance = self.filter.rank_point(intersection["boresight"])
            self.adjust_timestep(distance)
            if distance < self.rough_treshold:
                self._found_timestamps_cnt += 1
                return {
                    "et": self.current_simulation_timestamp_et,
                    "timestamp_utc": self.current_simulation_timestamp.to_datetime(),
                    "distance": distance,
                    "boresight": intersection["boresight"].tolist(),
                    "meta": {}
                }, None
            return None, None
        except Exception as e:
            # TODO: Persistent DB
            self._failed_timestamps_cnt += 1
            return None, {
                "et": self.current_simulation_timestamp_et,
                "timestamp_utc": self.current_simulation_timestamp.to_datetime(),
                "error": str(e),                
                "meta": {}
            }

    def start_simulation(self, start_time: Optional[Time] = None, end_time: Optional[Time] = None, batchsize: int = 1024):
        # Establish progress tracking with persistant database

        success_collection, failed_collection = Sessions._prepare_simulation_collections(self.instrument.name, self.filter.name)

        start_time = self.kernel_manager.min_loaded_time if start_time is None else start_time
        end_time = self.kernel_manager.max_loaded_time if end_time is None else end_time

        self._setup_time(time=start_time)

        positive_sensing_batch = []
        failed_computation_batch = []

        threads = []

        def check_threads():
            for thread_id in range(len(threads)-1, -1, -1):
                if not threads[thread_id].is_alive():
                    threads[thread_id].join()
                    threads.pop(thread_id)


        with tqdm(total=end_time - start_time, unit="time_step") as pbar:
            while self.current_simulation_timestamp < end_time:

                projection, failed_projection = self._calculate_state()

                if projection:
                    positive_sensing_batch.append(projection)
                if failed_projection:
                    failed_computation_batch.append(failed_projection)

                self._simulation_step()
                pbar.update(self.computation_timedelta.sec)

                if len(positive_sensing_batch) >= batchsize:
                    Sessions.background_insert_batch_timeseries_results(positive_sensing_batch, success_collection)
                    positive_sensing_batch = []
                    check_threads()
            
                if len(failed_computation_batch) >= batchsize:
                    Sessions.background_insert_batch_timeseries_results(failed_computation_batch, failed_collection)
                    failed_computation_batch = []
                    check_threads()

        # Since we want to kill all threads when the main thread exits, let's await them just to be sure!
        for thread in threads:
            thread.join()
