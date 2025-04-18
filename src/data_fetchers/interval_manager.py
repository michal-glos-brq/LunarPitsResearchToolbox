import heapq
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterator, Union

from astropy.time import Time, TimeDelta

import spiceypy as spice

from src.global_config import SPICE_DECIMAL_PRECISION


@dataclass
class TimeInterval:
    start_et: float  # Ephemeris time (seconds past J2000)
    end_et: float

    @property
    def start_astropy_time(self) -> Time:
        return Time(spice.et2utc(self.start_et, 'ISOC', SPICE_DECIMAL_PRECISION), format='isot', scale='utc')

    @property
    def end_astropy_time(self) -> Time:
        return Time(spice.et2utc(self.end_et, 'ISOC', SPICE_DECIMAL_PRECISION), format='isot', scale='utc')

    def __post_init__(self):
        if self.start_et >= self.end_et:
            raise ValueError("start_et must be less than end_et.")
        # Ensure that the interval is valid
        if self.start_et < 0 or self.end_et < 0:
            raise ValueError("Ephemeris time must be non-negative.")
        # Ensure that the interval is not too large
        if self.end_et - self.start_et > 1e9:
            raise ValueError("Interval is too large. Please check the values.")

    def is_interval_after(self, other: "TimeInterval") -> bool:
        """
        Check if the interval is after another interval.
        """
        return self.start_et > other.end_et

    def contains(self, et: float) -> bool:
        return self.start_et <= et < self.end_et

    def containes_time(self, time: Time) -> bool:
        return self.start_et <= spice.utc2et(time.utc.iso) < self.end_et

    def overlaps(self, other: "TimeInterval") -> bool:
        return self.start_et < other.end_et and self.end_et > other.start_et

    def is_subinterval(self, other: "TimeInterval") -> bool:
        return self.start_et >= other.start_et and self.end_et <= other.end_et


class IntervalList:

    @property
    def start_et(self) -> float:
        return self.intervals[0].start_et

    @property
    def end_et(self) -> float:
        return self.intervals[-1].end_et

    @property
    def start_astropy_time(self) -> Time:
        return self.intervals[0].start_astropy_time

    @property
    def end_astropy_time(self) -> Time:
        return self.intervals[-1].end_astropy_time

    def __init__(self, intervals: Union[List[TimeInterval], List[Tuple[float, float]]]):
        if not intervals:
            raise ValueError("Intervals list cannot be empty.")
        if isinstance(intervals[0], TimeInterval):
            self.intervals = intervals
        else:
            self.intervals = [TimeInterval(start_et, end_et) for start_et, end_et in intervals]
        self.intervals.sort(key=lambda x: x.start_et)
        # Linear access optimization: skips intervals before current query window
        self.linear_access_index = 0



    def get_intervals_intersection(self, interval: TimeInterval, linear_access=True) -> List[TimeInterval]:
        """
        Logical and on self.intervals and interval from parameters (Retuers subset of self.intervals, does not edit them,
        so it's not strict and, but includes only whole intervals from self.intervals).

        For some level of optimization, we can pass argument to assume linear access and when going
        for interval intersection, we start from the start of last intersection passed as parameter (or 0).
        """
        if not linear_access:
            return [iv for iv in self.intervals if iv.overlaps(interval)]
        # linear scan
        while (
            self.linear_access_index < len(self.intervals)
            and self.intervals[self.linear_access_index].end_et < interval.start_et
        ):
            self.linear_access_index += 1
        result: List[TimeInterval] = []
        idx = self.linear_access_index
        while idx < len(self.intervals) and self.intervals[idx].start_et < interval.end_et:
            if self.intervals[idx].overlaps(interval):
                result.append(self.intervals[idx])
            idx += 1
        self.linear_access_index = idx
        return result

    def intersection_mask(self, other: "IntervalList", linear_access: bool = True) -> List[bool]:
        """
        For each interval in this list, return a boolean indicating
        whether it overlaps any interval in `other`.
        """
        mask: List[bool] = []
        if linear_access:
            other.reset_linear_access()
        for iv in self.intervals:
            if linear_access:
                overlaps = other.get_intervals_intersection(iv, linear_access=True)
            else:
                overlaps = other.get_intervals_intersection(iv, linear_access=False)
            mask.append(bool(overlaps))
        return mask

    def reset_linear_access(self):
        """
        Reset the linear access index to the beginning.
        """
        self.linear_access_index = 0


class MultiIntervalQueue:
    """
    Merges intervals from multiple instruments into one global, sorted queue.
    Each entry is a tuple: (interval.start_et, instrument_name, interval)
    """

    def __init__(self, instrument_intervals: Dict[str, IntervalList]):
        self.heap = []
        for instrument, ilist in instrument_intervals.items():
            for interval in ilist.intervals:
                # Heap sorted by start_et
                heapq.heappush(self.heap, (interval.start_et, instrument, interval))

    def next_interval(self) -> Optional[Tuple[str, TimeInterval]]:
        if self.heap:
            _, instrument, interval = heapq.heappop(self.heap)
            return instrument, interval
        else:
            return None

    def __bool__(self) -> bool:
        return bool(self.heap)


class IntervalManager:
    """
    Manages time intervals for the simulation.
    Maintains active intervals for each instrument and provides a unified simulation time.
    """

    def __init__(self, intervals: Dict[str, list[Tuple[float, float]]]):
        self.intervals: Dict[str, IntervalList] = {
            instrument_name: IntervalList(intervals)
            for instrument_name, intervals in intervals.items()
        }
        self.multi_queue = MultiIntervalQueue(self.intervals)

    def get_interval_by_instrument(self, instrument_name: str) -> Optional[IntervalList]:
        """
        Get the interval list for a specific instrument.
        """
        return self.intervals.get(instrument_name)

    def next_interval(self) -> Optional[Tuple[str, TimeInterval]]:
        """
        Return the next interval tuple: (instrument_name, TimeInterval)
        from any instrument, in sorted order by start time.
        """
        return self.multi_queue.next_interval()

    def global_time_bounds(self) -> Tuple[float, float]:
        """
        Determine the overall global start and end times from all intervals.
        """
        min_start = min(ilist.start_et for ilist in self.intervals.values())
        max_end = max(ilist.end_et for ilist in self.intervals.values())
        return min_start, max_end

    def __bool__(self) -> bool:
        return bool(self.multi_queue)
