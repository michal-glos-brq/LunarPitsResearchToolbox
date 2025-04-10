import heapq
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterator

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

    def overlaps(self, other: "TimeInterval") -> bool:
        return self.start_et < other.end_et and self.end_et > other.start_et


class IntervalList:

    def __init__(self, instrument_name: str, intervals: List[Tuple[float, float]]):
        if not intervals:
            raise ValueError("Intervals list cannot be empty.")
        self.intervals = [TimeInterval(start_et, end_et) for start_et, end_et in intervals]
        self.instrument_name = instrument_name
        # Index into intervals for linear access - last overlapping interval from last intersection
        self.linear_access_index = 0

    def get_intervals_intersection(
        self, interval: TimeInterval, linear_access=True, sort_intervals: bool = False
    ) -> List[TimeInterval]:
        """
        Logical and on self.intervals and interval from parameters (Retuers subset of self.intervals, does not edit them,
        so it's not strict and, but includes only whole intervals from self.intervals).

        For some level of optimization, we can pass argument to assume linear access and when going
        for interval intersection, we start from the start of last intersection passed as parameter (or 0).
        """
        if sort_intervals:
            self.intervals.sort(key=lambda x: x.start_et)

        if linear_access:
            return [i for i in self.intervals if i.overlaps(interval)]
        else:
            # Find the first interval which starts at or after the interval's start_et
            while (
                self.linear_access_index < len(self.intervals)
                and self.intervals[self.linear_access_index].end_et < interval.start_et
            ):
                self.linear_access_index += 1

            result = []
            start_index = self.linear_access_index
            while start_index < len(self.intervals) and self.intervals[start_index].start_et < interval.end_et:
                if self.intervals[start_index].overlaps(interval):
                    result.append(self.intervals[start_index])
                start_index += 1
            self.linear_access_index = start_index
            return result


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

    def is_empty(self) -> bool:
        return len(self.heap) == 0


class IntervalManager:
    """
    Manages time intervals for the simulation.
    Maintains active intervals for each instrument and provides a unified simulation time.
    """

    def __init__(self, intervals: Dict[str, list[Tuple[float, float]]]):
        self.intervals: Dict[str, List[TimeInterval]] = {
            instrument_name: IntervalList(instrument_name, intervals)
            for instrument_name, intervals in intervals.items()
        }

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

    def is_empty(self) -> bool:
        return self.multi_queue.is_empty()

    def global_time_bounds(self) -> Tuple[float, float]:
        """
        Determine the overall global start and end times from all intervals.
        """
        min_start = min(
            interval.start_et for ilist in self.instrument_intervals.values() for interval in ilist.intervals
        )
        max_end = max(interval.end_et for ilist in self.instrument_intervals.values() for interval in ilist.intervals)
        return min_start, max_end
