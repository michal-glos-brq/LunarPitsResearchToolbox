import logging
from typing import Optional

from astropy.time import TimeDelta

from src.experiments.simulations import BaseSimulationConfig
from src.pipeline.app import run_remote_sensing_simulation_task
from src.pipeline.dispatchers.base_dispatcher import BaseTaskRunner


class RemoteSensingTaskRunner(BaseTaskRunner):
    """
    Task runner for remote sensing simulations.
    Splits the experiment time range into chunks and dispatches Celery tasks.
    """

    def run(self, config_name: str, dry_run: bool = True, simulation_name: str = None, retry_count: Optional[int] = None):

        config = BaseSimulationConfig.get_config_dict(config_name)

        start_time = config["start_time"]
        end_time = config["end_time"]
        step = TimeDelta(config["step_days"], format="jd")

        current_time = start_time
        task_counter = 0

        logging.info(f"Submitting tasks for experiment: {config_name}")
        while current_time < end_time:
            next_time = min(current_time + step, end_time)

            task_kwargs = dict(config["simulation_kwargs"])
            task_kwargs["start_time_et"] = current_time.cxcsec
            task_kwargs["end_time_et"] = next_time.cxcsec
            # It's possible to overrid the name assigned in simulation configuration via CLI parameters
            if simulation_name is not None:
                task_kwargs["simulation_name"] = simulation_name
            task_kwargs["retry_count"] = retry_count

            if not dry_run:
                result = run_remote_sensing_simulation_task.delay(**task_kwargs)
                logging.info(f"  Task {task_counter}: {current_time.iso} → {next_time.iso} | Task ID: {result.id}")
            else:
                logging.info(f"  Task {task_counter}: {current_time.iso} → {next_time.iso} | (Dry run)")

            current_time = next_time
            task_counter += 1

        logging.info(f"Submitted {task_counter} tasks successfully.")
