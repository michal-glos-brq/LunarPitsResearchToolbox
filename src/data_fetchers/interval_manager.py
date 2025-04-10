from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from astropy.time import Time, TimeDelta

import spiceypy as spice



@dataclass
class TimeInterval:
    start_et: float  # Ephemeris time (seconds past J2000)
    end_et: float

    def __post_init__(self):
        if self.start_et >= self.end_et:
            raise ValueError("start_et must be less than end_et.")
        # Ensure that the interval is valid
        if self.start_et < 0 or self.end_et < 0:
            raise ValueError("Ephemeris time must be non-negative.")
        # Ensure that the interval is not too large
        if self.end_et - self.start_et > 1e9:
            raise ValueError("Interval is too large. Please check the values.")

    def contains(self, et: float) -> bool:
        return self.start_et <= et < self.end_et

    def overlaps(self, other: 'TimeInterval') -> bool:
        return self.start_et < other.end_et and self.end_et > other.start_et


class IntervalList:

    def __init__(self, instrument_name: str, intervals: List[Tuple[float, float]]):
        if not intervals:
            raise ValueError("Intervals list cannot be empty.")
        self.intervals = [TimeInterval(start_et, end_et) for start_et, end_et in intervals]
        self.instrument_name = instrument_name
        # Index into intervals for linear access - last overlapping interval from last intersection
        self.linear_access_index = 0


    def get_intervals_intersection(self, interval: TimeInterval, linear_access = True) -> List[TimeInterval]:
        """
        Logical and on self.intervals and interval from parameters (Retuers subset of self.intervals, does not edit them,
        so it's not strict and, but includes only whole intervals from self.intervals).
        
        For some level of optimization, we can pass argument to assume linear access and when going
        for interval intersection, we start from the start of last intersection passed as parameter (or 0).
        """
        if linear_access:
            # We need to check this customly, because it might not interlap, but start after and the linear iteration would be shattered
            # Find the first interval which starts after (or at the same time) as the interval's start_et
            while self.intervals[self.linear_access_index].start_et <= interval.start_et:
                self.linear_access_index += 1
                if self.linear_access_index >= len(self.intervals):
                    return []

            # In case the found interval starts after the intersected interval ends, return empty list
            if self.intervals[self.linear_access_index].start_et > interval.end_et:
                self.linear_access_index += 1
                return []

            intervals = []
            for i in self.intervals[self.linear_access_index:]:
                if i.end_et < interval.start_et:
                    continue
                elif i.start_et > interval.end_et:
                    break
                else:
                    intervals.append(i) 


        else:    
            return [i for i in self.intervals if i.overlaps(interval)]


class IntervalManager:
    """
    Manages time intervals for the simulation.
    Maintains active intervals for each instrument and provides a unified simulation time.
    """
    def __init__(self, intervals: Dict[str, list[Tuple[float, float]]]):
        self.intervals: Dict[str, List[TimeInterval]] = {instrument_name: IntervalList(instrument_name, intervals) for instrument_name, intervals in intervals.items()}


    def get_interval_by_instrument(self, instrument_name: str) -> Optional[IntervalList]:
        """
        Get the interval list for a specific instrument.
        """
        return self.intervals.get(instrument_name)


    def next(self) -> Optional[Time]:
        """
        Advance simulation time by one step.
        Returns the new simulation time, or None if past global_end.
        """
        if self.current_time < self.global_end:
            self.current_time += self.step
            return self.current_time
        else:
            return None

    def current_et(self) -> float:
        """
        Returns current simulation time in ephemeris time.
        """
        return spice.str2et(self.current_time.utc.iso)