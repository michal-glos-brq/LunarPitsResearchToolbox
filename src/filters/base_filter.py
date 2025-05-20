"""
============================================================
Base filter implementation
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseFilter(ABC):

    _hard_radius = 0.0

    @property
    def hard_radius(self) -> float:
        # How far from out area or point of interest are we capturing the data
        return self._hard_radius

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def _name(**kwargs) -> str:
        pass

    @abstractmethod
    def rank_point(self, point: np.array) -> float:
        pass

    @classmethod
    @abstractmethod
    def from_kwargs_and_kernel_manager(cls, kernel_manager, **kwargs):
        """
        This method is used to create a filter instance from the given kernel manager and keyword arguments.
        It extracts the DSK filename from the kernel manager and uses it to initialize the filter.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
