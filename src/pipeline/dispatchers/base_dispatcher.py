"""
============================================================
Base dispatcher class which assigns tasks to celery broker
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

from abc import ABC, abstractmethod


class BaseTaskRunner(ABC):
    """
    Abstract base class for task runners.
    """

    @abstractmethod
    def run(self, config_name: str, dry_run: bool = True):
        """
        Run the task with the given experiment config name.
        """
        pass
