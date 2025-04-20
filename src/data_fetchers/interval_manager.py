import heapq
import logging
from bisect import bisect_right
from dataclasses import dataclass
from functools import total_ordering
from typing import List, Dict, Optional, Tuple, Union, Any

from astropy.time import Time
import spiceypy as spice

from src.SPICE.utils import et2astropy_time
from src.global_config import SPICE_DECIMAL_PRECISION


logger = logging.getLogger(__name__)


@total_ordering
@dataclass
class TimeInterval:
    start_et: float  # Ephemeris time (seconds past J2000)
    end_et: float

    def to_json(self) -> Dict[str, float]:
        """JSON‑friendly dict."""
        return {"start_et": self.start_et, "end_et": self.end_et}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "TimeInterval":
        """Reconstruct from to_json output."""
        return cls(data["start_et"], data["end_et"])

    @property
    def start_astropy_time(self) -> Time:
        return et2astropy_time(self.start_et)

    @property
    def end_astropy_time(self) -> Time:
        return et2astropy_time(self.end_et)

    def __post_init__(self):
        if self.start_et >= self.end_et:
            raise ValueError("start_et must be less than end_et.")
        # Ensure that the interval is valid
        if self.start_et < 0 or self.end_et < 0:
            raise ValueError("Ephemeris time must be non-negative.")
        # Ensure that the interval is not too large
        if self.end_et - self.start_et > 1e9:
            raise ValueError("Interval is too large. Please check the values.")

    def __contains__(self, other: object) -> bool:
        if isinstance(other, float):
            return self.start_et <= other < self.end_et
        elif isinstance(other, Time):
            return self.start_et <= spice.utc2et(other.iso) < self.end_et
        elif isinstance(other, TimeInterval):
            return self.start_et <= other.start_et and self.end_et >= other.end_et
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TimeInterval):
            return (self.start_et, self.end_et) == (other.start_et, other.end_et)
        return NotImplemented

    def __lt__(self, other: "TimeInterval") -> bool:
        if self.start_et < other.start_et:
            return True
        elif self.start_et == other.start_et:
            return self.end_et < other.end_et
        else:
            return False

    def is_interval_after(self, other: "TimeInterval") -> bool:
        """Check if the whole interval (self) is after another (other) interval."""
        return self.start_et > other.end_et

    def is_interval_before(self, other: "TimeInterval") -> bool:
        """Check if the whole interval (self) is before another (other) interval."""
        return self.end_et < other.start_et

    def overlaps(self, other: "TimeInterval") -> bool:
        return not (self.end_et < other.start_et or other.end_et < self.start_et)

    def __repr__(self) -> str:
        return f"TimeInterval({self.start_et}, {self.end_et})"

    def __and__(self, other: "TimeInterval") -> "TimeInterval":
        if not self.overlaps(other):
            return None
        return TimeInterval(max(self.start_et, other.start_et), min(self.end_et, other.end_et))

    def __or__(self, other: "TimeInterval") -> "TimeInterval":
        if not self.overlaps(other):
            return None
        return TimeInterval(min(self.start_et, other.start_et), max(self.end_et, other.end_et))


class IntervalList:

    def __init__(self, intervals: Union[List[TimeInterval], List[Tuple[float, float]]]):
        if not intervals:
            raise ValueError("Intervals list cannot be empty.")

        self.intervals = (
            intervals
            if isinstance(intervals[0], TimeInterval)
            else [TimeInterval(start_et, end_et) for start_et, end_et in intervals]
        )
        self.intervals.sort()
        # precompute sorted end times for bisect
        self._ends = [iv.end_et for iv in self.intervals]

    def to_json(self) -> List[Dict[str, float]]:
        """List of interval dicts."""
        return [iv.to_json() for iv in self.intervals]

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]]) -> "IntervalList":
        """Reconstruct from to_json output."""
        ivs = [TimeInterval.from_json(d) for d in data]
        return cls(ivs)

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

    def get_intervals_intersection(self, target: TimeInterval) -> List[TimeInterval]:
        """
        Return all intervals in self.intervals that overlap `target`.
        Runs in O(log n + k) time where k is number of overlaps.
        """
        # find first interval whose end > target.start_et
        i = bisect_right(self._ends, target.start_et)
        results = []
        for iv in self.intervals[i:]:
            if iv.start_et >= target.end_et:
                break
            # by construction iv.end_et > target.start_et and iv.start_et < target.end_et
            results.append(iv)
        return results

    def intersection_mask(self, other: "IntervalList") -> List[bool]:
        """
        For each interval in self.intervals, True if it overlaps ANY in `other`.
        """
        # ensure other's ends array is up-to-date
        other_ends = [iv.end_et for iv in other.intervals]

        mask = []
        for iv in self.intervals:
            # find the first other interval that might overlap
            j = bisect_right(other_ends, iv.start_et)
            overlap = False
            for oiv in other.intervals[j:]:
                if oiv.start_et >= iv.end_et:
                    break
                overlap = True
                break
            mask.append(overlap)
        return mask

    def __repr__(self) -> str:
        return f"IntervalList({self.intervals})"


class MultiIntervalQueue:
    """
    Merges intervals from multiple instruments into one global, sorted queue.
    Each entry is a tuple: (interval.start_et, instrument_name, interval)
    """

    def __init__(self, instrument_intervals: Dict[str, IntervalList]):
        self.heap: List[Tuple[TimeInterval, str]] = [
            (iv, name) for name, ilist in instrument_intervals.items() for iv in ilist.intervals
        ]
        heapq.heapify(self.heap)

    def next_interval(self) -> Optional[Tuple[str, TimeInterval]]:
        if not self.heap:
            return None
        iv, name = heapq.heappop(self.heap)
        return name, iv

    def __bool__(self) -> bool:
        return bool(self.heap)

    def __len__(self) -> int:
        return len(self.heap)

class IntervalManager:
    """
    Manages time intervals for the simulation.
    Maintains active intervals for each instrument and provides a unified simulation time.
    """

    def __init__(self, intervals: Dict[str, list[Tuple[float, float]]]):
        self.intervals: Dict[str, IntervalList] = {
            instrument_name: IntervalList(intervals) for instrument_name, intervals in intervals.items()
        }
        self.multi_queue = MultiIntervalQueue(self.intervals)

    def to_json(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Serialize to a JSON‑friendly dict:
          { instrument_name: [ {start_et:…, end_et:…}, … ], … }
        """
        return {name: ilist.to_json() for name, ilist in self.intervals.items()}

    @classmethod
    def from_json(cls, data: Dict[str, List[Dict[str, float]]]) -> "IntervalManager":
        """
        Reconstruct an IntervalManager from its to_json output.
        """
        raw = {name: [(d["start_et"], d["end_et"]) for d in lst] for name, lst in data.items()}
        return cls(raw)

    def split_by_times(self, et_points: List[float]) -> List["IntervalManager"]:
        """
        Slice this manager into consecutive sub‐managers at the given ET points.

        - If `et_points` is empty or none of the points fall inside
          this manager’s global [start, end), returns [self].
        - Otherwise, returns N+1 managers, where N = number of valid split points:
            [start, p1), [p1, p2), …, [pN, end).

        Each sub‐manager will only contain the portions of original intervals
        that lie within its slice.

        Args:
            et_points: arbitrary list of split‐times (floats). They needn’t be sorted.
        Returns:
            List of IntervalManager, in ascending time order.
        """
        # 1) compute global bounds
        global_start, global_end = self.global_time_bounds()

        # 2) keep only points strictly inside (global_start, global_end)
        pts = sorted({p for p in et_points if global_start < p < global_end})
        if not pts:
            return [self]

        # 3) build the full boundary list: [start, p1, p2, …, pN, end]
        boundaries = [global_start] + pts + [global_end]

        managers: List[IntervalManager] = []
        for a, b in zip(boundaries, boundaries[1:]):
            slice_iv = TimeInterval(a, b)
            sub_intervals: Dict[str, List[Tuple[float, float]]] = {}

            # 4) for each instrument, intersect its intervals with [a,b)
            for name, ilist in self.intervals.items():
                # fast O(log n + k) to get overlapping ivs
                for iv in ilist.get_intervals_intersection(slice_iv):
                    cut = iv & slice_iv  # guaranteed non‐None here
                    sub_intervals.setdefault(name, []).append((cut.start_et, cut.end_et))

            # 5) even if an instrument has no coverage, we include it only if any instrument did
            managers.append(IntervalManager(sub_intervals))

        return managers

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

    def __len__(self) -> int:
        return len(self.multi_queue)

    def __bool__(self) -> bool:
        return bool(self.multi_queue)

    def __repr__(self) -> str:
        """
        Show each instrument’s list of intervals in one line.
        """
        parts = ", ".join(f"{name}: {ilist}" for name, ilist in self.intervals.items())
        return f"IntervalManager({{{parts}}})"
