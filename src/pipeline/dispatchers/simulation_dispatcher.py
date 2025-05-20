"""
============================================================
Dispatcher 
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""

import logging
from typing import Optional

from astropy.time import TimeDelta

from src.experiments.simulations.lunar_pit_simulation import BaseSimulationConfig
from src.pipeline.app import run_remote_sensing_simulation_task
from src.pipeline.dispatchers.base_dispatcher import BaseTaskRunner

logger = logging.getLogger(__name__)

class RemoteSensingTaskRunner(BaseTaskRunner):
    """
    Task runner for remote sensing simulations.
    Splits the experiment time range into chunks and dispatches Celery tasks.
    """

    def run(self, config_name: str, dry_run: bool = True, name: str = None, retry_count: Optional[int] = None):

        config = BaseSimulationConfig.get_config_dict(config_name)

        start_time = config["start_time"]
        end_time = config["end_time"]
        step = TimeDelta(config["step_days"], format="jd")

        current_time = start_time
        task_counter = 0

        logger.info(f"Submitting tasks for experiment: {config_name}")
        while current_time < end_time:
            next_time = min(current_time + step, end_time)

            task_kwargs = dict(config["simulation_kwargs"])
            task_kwargs["start_time_isot"] = current_time.isot
            task_kwargs["end_time_isot"] = next_time.isot

            if name is not None:
                task_kwargs["simulation_name"] = name
            task_kwargs["retry_count"] = retry_count

            if not dry_run:
                result = run_remote_sensing_simulation_task.delay(**task_kwargs)
                logger.info(f"  Task {task_counter}: {current_time.iso} → {next_time.iso} | Task ID: {result.id}")
            else:
                logger.info(f"  Task {task_counter}: {current_time.iso} → {next_time.iso} | (Dry run)")

            current_time = next_time
            task_counter += 1

        logger.info(f"Submitted {task_counter} tasks successfully.")
