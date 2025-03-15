import sys
import logging
from scipy.spatial import cKDTree
from tqdm import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import spiceypy as spice
from astropy.time import Time, TimeDelta


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


sys.path.insert(0, "/".join(__file__.split("/")[:-4]))
from src.SPICE.config import (
    MOON_STR_ID,
    ABBERRATION_CORRECTION,
    MOON_REF_FRAME_STR_ID,
    TIME_STEP,
    MAX_TIME_STEP,
    LRO_SPEED,
    LUNAR_MODEL,
)
from src.global_config import TQDM_NCOLS
from src.db.mongo.interface import Sessions


class HandledExpeption(Exception):
    pass



class Instrument(ABC):

    @property
    @abstractmethod
    def sweep_iterator_class(self): ...
 
    @property
    @abstractmethod
    def subinstrumen_offset(self) -> float: ...

    @property
    @abstractmethod
    def fov_offset(self) -> float: ...

    @property
    @abstractmethod
    def frame(self) -> str: ...

    @property
    @abstractmethod
    def distance_tolerance(self) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def instrument_ids(self) -> str: ...

    @property
    @abstractmethod
    def satellite_frame(self) -> str: ...

    @property
    @abstractmethod
    def offset_days(self) -> float: ...

    @dataclass
    class SubInstrument:
        _id: int
        frame: str
        bounds: np.array
        boresight: np.array


    # This can be set and when is - acts as override value of subinstrument frames and boresights + bounds are transformed to this frame
    _universal_subinstrument_frame = None
    # Here we aggregate the mean of subinstrument boresight in 
    _boresight = None

    @property
    def boresight(self):
        if self._boresight is None:
            self._boresight = np.stack([sub.boresight for sub in self.sub_instruments.values()]).mean(axis=0)
        return self._boresight

    @property
    def min_time(self):
        return self.sweep_iterator.min_loaded_time

    @property
    def max_time(self):
        return self.sweep_iterator.max_loaded_time

    @property
    def sub_instrument_frames(self) -> List[str]:
        return [sub_instrument.frame for sub_instrument in self.sub_instruments.values()]

    @property
    def rough_treshold(self) -> float:
        return self.subinstrumen_offset + self.distance_tolerance + self.fov_offset

    @property
    def finer_treshold(self) -> float:
        return self.distance_tolerance


    def __init__(self):
        self.sweep_iterator = self.sweep_iterator_class()
        self.computation_timedelta = TimeDelta(TIME_STEP, format="sec")
        self.current_simulation_timestamp = self.sweep_iterator.min_loaded_time + self.computation_timedelta
        self.current_simulation_timestamp_et = spice.str2et(self.current_simulation_timestamp.utc.iso)
        self.current_simulation_step = 0

        # Get points of interest and build a KD-Tree for fast spatial searches
        self._load_target_points()

        # Set simulation start time considering instrument offset
        self._set_time(self.sweep_iterator.min_loaded_time + TimeDelta(self.offset_days, format="jd"))
        self.sweep_iterator.initiate_sweep(self.current_simulation_timestamp)

        self.sub_instruments: Dict[int, Tuple[str, np.ndarray, np.ndarray]] = {}
        self.instantiate_subinstruments()
        # Setting the uniform frame for all sub-instruments to skip on tranformation matrix computation for each subinstrument
        self.uniform_sub_instrument_frame = list(self.sub_instruments.values())[0].frame if self._universal_subinstrument_frame is None else self._universal_subinstrument_frame
        if len(set(self.sub_instrument_frames)) == 1 and self._universal_subinstrument_frame is not None:
            self._transformation_matrix = (-1, None)
        else:
            # Convert the rest of the sensors into the same frame
            for sub_instrument in self.sub_instruments.values():
                projection_matrix = spice.pxform(sub_instrument.frame, self.uniform_sub_instrument_frame, self.current_simulation_timestamp_et)
                sub_instrument.boresight = spice.mxv(projection_matrix, sub_instrument.boresight)
                sub_instrument.bounds = np.array([spice.mxv(projection_matrix, bound) for bound in sub_instrument.bounds])
                sub_instrument.frame = self.uniform_sub_instrument_frame

        # Tracking lists
        self._found_timestamps, self._found_timestamps_cnt = [], 0
        self._boresight_projections = []
        self._failed_timestamps, self._failed_timestamps_cnt = [], 0
        self.adjusted_timesteps = []
        self.min_distances = []


    def instantiate_subinstruments(self):
        for naif_id in self.instrument_ids:
            # Get FOV shape, boresight, and boundary vectors
            _, frame, boresight, _, bounds = spice.getfov(naif_id, room=1000)
            self.sub_instruments[naif_id] = self.SubInstrument(naif_id, frame, np.array(bounds), np.array(boresight))


    def project_vector(self, et, vector) -> np.array:
        # spice.sincpt("DSK/UNPRIORITIZED", MOON_STR_ID, et, MOON_REF_FRAME_STR_ID, ABBERRATION_CORRECTION, self.satellite_frame, self.frame, vector)
        # import pdb; pdb.set_trace()
        return spice.sincpt(
            "DSK/UNPRIORITIZED",
            MOON_STR_ID,
            et,  # Time (just a number, the astro time)
            MOON_REF_FRAME_STR_ID,
            ABBERRATION_CORRECTION,
            self.satellite_frame,
            self.frame,
            vector,
        )

    def transformation_matrix(self, et) -> Optional[np.array]:
        if et == self._transformation_matrix[0]:
            return self._transformation_matrix[1]
        else:
            matrix = spice.pxform(self.uniform_sub_instrument_frame, self.frame, et)
            self._transformation_matrix = (et, matrix)
            return matrix

    def compute_views_instrument_boresight(self, et) -> Dict[str, Dict]:
        """
        Compute views for the instrument at given time
        Where on the Lunar surface are we looking at
        """
        boresight_point, boresight_trgepc, _ = self.project_vector(
            et, spice.mxv(self.transformation_matrix(et), self.boresight)
        )
        return {"et": et, "boresight": boresight_point, "boresight_trgepc": boresight_trgepc}

    def compute_views_subinstruments_boresight(self, et) -> Dict[int, Dict]:
        """
        Compute views for the instrument at given time
        Will project the point in
        """
        # Here, we store ProjectedPoint for each subinstrument
        boresights = {}

        for sub_instrument in self.sub_instruments.values():
            boresight_point, boresight_trgepc, _ = self.project_vector(
                et, spice.mxv(self.transformation_matrix(et), sub_instrument.boresight)
            )
            boresights[sub_instrument._id] = {
                "et": et,
                "boresight": boresight_point,
                "boresight_trgepc": boresight_trgepc,
            }
        return boresights

    def compute_views_subinstruments_bounds(self, et) -> Dict[int, Dict]:
        """
        Compute views for the instrument at given time
        Will project the point in
        """
        # Here, we store ProjectedPoint for each subinstrument
        bounds = {}

        for sub_instrument in self.sub_instruments.values():
            bound_points, bound_trgpecs = [], []
            for bound in sub_instrument.bounds:
                bound_point, bound_trgpec, _ = self.project_vector(et, spice.mxv(self.transformation_matrix(et), bound))
                bound_points.append(bound_point)
                bound_trgpecs.append(bound_trgpec)
            bounds[sub_instrument._id] = {
                "et": et,
                "bounds": bound_points,
                "bounds_trgepc": bound_trgpecs,
            }
        return bounds

    def _load_target_points(self):
        """Fetches crater points and converts lat/lon to Cartesian coordinates using DSK."""
        points = Sessions.get_all_pits_points()
        lat_rad = np.radians(points["latitude"])
        lon_rad = np.radians(points["longitude"])

        # Load DSK file and get a valid handle
        dsk_handle = spice.dasopr(LUNAR_MODEL['dsk_path'])  # Open the DSK file
        # Find the DSK segment descriptor
        dladsc = spice.dlabfs(dsk_handle)  # Get the first segment in the file

        cartesian_points = []
        for lat, lon in zip(lat_rad, lon_rad):
            # Convert lat/lon to a unit vector for the ray direction
            cartesians = np.array(spice.latrec(1.0, lon, lat))  # Ensure it's an array

            # We have to simulate observer above the ground, looking at [0,0,0], because interception could not be calculated from within
            _, spoint, found = spice.dskx02(dsk_handle, dladsc, cartesians * 10_000, (-1) * cartesians)

            if found:
                cartesian_points.append(spoint)
            else:
                logger.warning(f"No surface intercept found for lat: {lat}, lon: {lon}")
                cartesian_points.append([np.nan, np.nan, np.nan])  # Mark missing points

        # Close the DSK file
        spice.dascls(dsk_handle)

        # Store computed points
        points["X"], points["Y"], points["Z"] = np.array(cartesian_points).T
        self._target_points = points[["X", "Y", "Z"]].values
        self._target_ids = np.arange(len(self._target_points))
        self.kd_tree = cKDTree(self._target_points)



    def _step_time(self):
        self.current_simulation_timestamp += self.computation_timedelta
        self.current_simulation_timestamp_et = spice.str2et(self.current_simulation_timestamp.utc.iso)
        self.current_simulation_step += 1
        self.sweep_iterator.step(self.current_simulation_timestamp)

    def _set_time(self, time: Time, timestep: Optional[int] = None):
        self.current_simulation_timestamp = time
        self.current_simulation_timestamp_et = spice.str2et(time.utc.iso)
        self.current_simulation_step = 0 if timestep is None else timestep
        self.sweep_iterator.step(self.current_simulation_timestamp)


    def adjust_timestep(self, min_distance: float):
        new_time_step = (min_distance - self.rough_treshold) / LRO_SPEED
        self.computation_timedelta = TimeDelta(min(max(TIME_STEP, new_time_step), MAX_TIME_STEP), format="sec")
        self.adjusted_timesteps.append(new_time_step)
        self.min_distances.append(min_distance)

    def simulation_step_inference(self):
        try:
            #import pdb; pdb.set_trace()
            # Projects to the lunar surface and looks for closest points (may be empty)
            intersection = self.compute_views_instrument_boresight(self.current_simulation_timestamp_et)
            boresight = intersection["boresight"]
            min_distance = self.kd_tree.query(boresight)[0]
            self.adjust_timestep(min_distance)
            if min_distance < self.rough_treshold:
                self._found_timestamps_cnt += 1
                return {
                    "instrument": self.name,
                    "et": self.current_simulation_timestamp_et,
                    "timestamp_utc": self.current_simulation_timestamp.to_datetime(),
                    "min_distance": min_distance,
                    "boresight": boresight.tolist(),
                    "meta": {}
                }
        except Exception as e:
            #self._failed_timestamps.append((self.current_simulation_timestamp, self.current_simulation_step))
            self._failed_timestamps_cnt += 1
            raise HandledExpeption(e)


    def run_simulation(self, max_steps: Optional[int] = None, collection_slug: Optional[str] = None):
        # Prepare misc
        total_seconds = (self.max_time - self.min_time).to_value("sec")
        pbar_format_string = ("" if max_steps is None else f"/{max_steps}")

        # Here we store our points of interest to dump them into DB, eventually
        points_of_interest_batch = []


        simulation_collection = Sessions._prepare_simulation_collection(collection_slug)
        # Run the main simulation loop
        with tqdm(total=total_seconds, ncols=TQDM_NCOLS, desc="Running simulation") as pbar:

            while self.current_simulation_timestamp <= self.max_time:
            
                # TQDM instrumentation
                pbar.update(self.computation_timedelta.to_value("sec"))
                if self.current_simulation_step % 256 == 0:
                    step_info = f"Simulation step:{self.current_simulation_step}"
                    failed_found_info = f"; failed: {self._failed_timestamps_cnt}; found: {self._found_timestamps_cnt}"
                    pbar.set_description(step_info + pbar_format_string + failed_found_info)

                # Check if we reached the maximum number of steps
                if max_steps is not None and self.current_simulation_step >= max_steps:
                    break
                
                try:
                    self._step_time()
                    if (simulation_step_output := self.simulation_step_inference()) is not None:
                        points_of_interest_batch.append(simulation_step_output)
                except HandledExpeption as e:
                    pass
                except Exception as e:
                    self._failed_timestamps_cnt += 1

                if len(points_of_interest_batch) > 1000:
                    Sessions.insert_simulation_results(points_of_interest_batch, collection=simulation_collection)
                    points_of_interest_batch = []

            # Add the last batch of points
            Sessions.insert_simulation_results(points_of_interest_batch, collection=simulation_collection)

        return

