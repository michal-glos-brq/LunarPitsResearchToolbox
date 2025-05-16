"""
============================================================
Dispatcher for extraction celery tasks
============================================================

Author: Michal Glos
University: Brno University of Technology (VUT)
Faculty: Faculty of Electrical Engineering and Communication (FEKT)
Diploma Thesis Project
"""


import logging
from typing import Optional

import spiceypy as spice
from astropy.time import TimeDelta

from src.structures import IntervalManager
from src.experiments.extractions.lunar_pit_data_extraction import BaseExtractionConfig
from src.pipeline.app import run_data_extraction_task
from src.pipeline.dispatchers.base_dispatcher import BaseTaskRunner
from src.db.interface import Sessions
from src.SPICE.kernel_utils.spice_kernels import BaseKernel
from src.SPICE.config import root_path

logger = logging.getLogger(__name__)


class ExtractorTaskRunner(BaseTaskRunner):
    """
    Task runner for remote sensing simulations.
    Splits the experiment time range into chunks and dispatches Celery tasks.
    """

    def run(
        self,
        config_name: str,
        dry_run: bool = True,
        name: str = None,
        retry_count: Optional[int] = None,
    ):
        config = BaseExtractionConfig.get_config_dict(config_name)

        start_time = config["start_time"]
        end_time = config["end_time"]
        step = TimeDelta(config["step_days"], format="jd")

        leapseconds_kernel = BaseKernel(
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls", root_path("lsk/naif0012.tls")
        )
        leapseconds_kernel.load()

        current_time = start_time
        task_counter = 0

        logger.info(f"Splitting time into task chunks ...")

        time_list = []
        current_time = start_time
        while current_time < end_time:
            time_list.append(current_time)
            current_time += step
        time_list.append(end_time)  # Ensure inclusive end

        et_list = [spice.utc2et(t.iso) for t in time_list]

        logger.info("Obtaining intervals from the DB ...")
        extracxtion_kwargs = config["extraction_kwargs"]
        intervals = Sessions.get_simulation_intervals(extracxtion_kwargs["instrument_names"], config["interval_name"])

        logger.info("Initializing interval managers ...")
        interval_manager = IntervalManager(intervals)
        interval_managers = interval_manager.split_by_timestamps(et_list)

        logger.info(
            f"Submitting tasks for extraction: {config['experiment_name']}; run {name}; intervals: {config['interval_name']}"
        )

        for _interval_manager in interval_managers:

            if not _interval_manager:
                logger.info("Empty interval manager, skipping...")
                continue

            task_kwargs = dict(config["extraction_kwargs"])

            task_kwargs["start_time_isot"] = _interval_manager.start_astropy_time.isot
            task_kwargs["end_time_isot"] = _interval_manager.end_astropy_time.isot
            task_kwargs["time_interval_manager_json"] = _interval_manager.to_json()

            task_kwargs["extraction_name"] = name
            task_kwargs["retry_count"] = retry_count

            if not dry_run:
                result = run_data_extraction_task.delay(**task_kwargs)
                logger.info(
                    f"  Task {task_counter}: {_interval_manager.start_astropy_time.isot} → {_interval_manager.end_astropy_time.isot} | Task ID: {result.id}"
                )
            else:
                logger.info(
                    f"  Task {task_counter}: {_interval_manager.start_astropy_time.isot} → {_interval_manager.end_astropy_time.isot} | (Dry run)"
                )

            task_counter += 1

        logger.info(f"Submitted {task_counter} tasks successfully.")
