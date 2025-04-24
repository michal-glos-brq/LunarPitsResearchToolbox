from .base_filter import BaseFilter
from .area_filter import AreaFilter
from .point_filter import PointFilter
from .lunar_pit_filter import LunarPitFilter
from .composite_filter import CompositeFilter


__all__ = [
    "BaseFilter",
    "AreaFilter",
    "PointFilter",
    "LunarPitFilter",
    "CompositeFilter",
]

FILTER_MAP = {
    "point": PointFilter,
    "area": AreaFilter,
    "lunar_pit": LunarPitFilter,
    "composite": CompositeFilter,
}
