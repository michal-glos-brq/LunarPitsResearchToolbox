from abc import ABC, abstractmethod


class BaseTaskRunner(ABC):
    """
    Abstract base class for task runners.
    """

    @abstractmethod
    def run(self, config_name: str):
        """
        Run the task with the given experiment config name.
        """
        pass
