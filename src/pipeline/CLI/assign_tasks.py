import argparse
import logging
from dataclasses import dataclass

from src.experiments.simulations import BaseSimulationConfig
from src.pipeline.dispatchers.base_dispatcher import BaseTaskRunner
from src.pipeline.dispatchers.simulation_dispatcher import RemoteSensingTaskRunner


@dataclass
class Task:
    """Declaration of base class of config and runner"""

    config_class: type
    runner_class: type


TASK_CLASSES = {
    "remote_sensing": Task(RemoteSensingTaskRunner, BaseSimulationConfig),
}


@dataclass
class TaskConfig:
    """
    Configuration for the task runner.
    """

    task_type: str
    config_name: str

    @property
    def task_runner_class(self):
        return TASK_CLASSES[self.task_type].runner_class

    @property
    def config_class(self):
        return TASK_CLASSES[self.task_type].config_class


def main():
    parser = argparse.ArgumentParser(description="Task runner for lunar simulation experiments.")
    parser.add_argument(
        "task", choices=["remote_sensing"], help="Type of task to run (currently only 'remote_sensing' is supported)."
    )
    parser.add_argument("config_name", help="Name of the experiment config to use.")
    args = parser.parse_args()

    try:
        task_config = TaskConfig(task_type=args.task, config_name=args.config_name)
        runner_cls = get_task_runner_class(args.task)
        runner = runner_cls()
        runner.run(args.config_name)
    except Exception as e:
        logging.info(f"Error while running task: {e}")
        list_available_configs()


if __name__ == "__main__":
    main()
