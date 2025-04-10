#! /usr/bin/env python3

import argparse
import logging
from dataclasses import dataclass

from src.experiments.simulations import BaseSimulationConfig
from src.pipeline.dispatchers.simulation_dispatcher import RemoteSensingTaskRunner


@dataclass
class Task:
    """Declaration of base class of config and runner"""

    config_class: type
    runner_class: type


TASK_CLASSES = {
    # src/experiments/simulations.py - contains allconfigurations for this task
    "remote_sensing": Task(BaseSimulationConfig, RemoteSensingTaskRunner),
}


@dataclass
class TaskConfig:
    """
    Configuration for the task runner.
    """

    task_type: str
    config_name: str

    def __post_init__(self):
        if self.task_type not in TASK_CLASSES:
            raise ValueError(f"Unknown task type '{self.task_type}'. Available tasks: {list(TASK_CLASSES.keys())}")

        if self.config_name not in TASK_CLASSES[self.task_type].config_class.registry:
            available = list(TASK_CLASSES[self.task_type].config_class.registry.keys())
            raise ValueError(f"Unknown experiment config '{self.config_name}'. Available configs: {available}")

    @property
    def task_runner_class(self):
        return TASK_CLASSES[self.task_type].runner_class

    @property
    def config_class(self):
        return TASK_CLASSES[self.task_type].config_class


def main():
    parser = argparse.ArgumentParser(description="Task runner for lunar simulation experiments.")
    parser.add_argument(
        "--task", choices=["remote_sensing"], help="Type of task to run (currently only 'remote_sensing' is supported)."
    )
    parser.add_argument("--config-name", help="Name of the experiment config to use.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run in dry run mode (no actual tasks will be submitted)."
    )
    parser.add_argument("--name", help="Name of the experiment config to use.")
    parser.add_argument("--retry-count", type=int, default=None, help="Indication whether the run is a retry run.")
    args = parser.parse_args()

    try:
        task_config = TaskConfig(task_type=args.task, config_name=args.config_name)
        runner = task_config.task_runner_class()
        logging.info(f"Running task: {args.task} with config '{args.config_name}'")
        runner.run(args.config_name, dry_run=args.dry_run, simulation_name=args.name, retry_count=args.retry_count)

    except Exception as e:
        logging.info(f"Error while running task: {e}")
        raise e


if __name__ == "__main__":
    main()
