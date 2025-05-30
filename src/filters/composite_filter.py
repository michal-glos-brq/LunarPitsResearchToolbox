"""
============================================================
Composite Filter implementation
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

import numpy as np

from .base_filter import BaseFilter


class CompositeFilter(BaseFilter):
    """This filter is used to combine multiple filters"""

    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters

    @property
    def name(self) -> str:
        return f"CompositeFilter_{'_'.join([f.name for f in self.filters])}"

    def rank_point(self, point: np.array) -> float:
        return min([f.rank_point(point) for f in self.filters])
