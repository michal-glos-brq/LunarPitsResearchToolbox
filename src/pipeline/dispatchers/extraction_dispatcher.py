import logging
from typing import Optional

import spiceypy as spice
from astropy.time import TimeDelta

from src.data_fetchers.interval_manager import IntervalManager
from src.experiments.extractions import BaseExtractionConfig
from src.pipeline.app import run_data_extraction_task
from src.pipeline.dispatchers.base_dispatcher import BaseTaskRunner
from src.db.interface import Sessions
from src.SPICE.kernel_utils.spice_kernels import BaseKernel
from src.SPICE.config import root_path


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

        logging.info(f"Splitting time into task chunks ...")

        time_list = []
        current_time = start_time
        while current_time < end_time:
            time_list.append(current_time)
            current_time += step
        time_list.append(end_time)  # Ensure inclusive end

        et_list = [spice.utc2et(t.iso) for t in time_list]

        logging.info("Obtaining intervals from the DB ...")
        intervals = Sessions.get_simulation_intervals(config.instrument_names, config.interval_name)

        logging.info("Initializing interval managers ...")
        interval_manager = IntervalManager(intervals)
        interval_managers = interval_manager.split_by_times(et_list)

        logging.info(
            f"Submitting tasks for extraction: {config.experiment_name}; run {name}; intervals: {config.interval_name}"
        )

        assert (
            len(interval_managers) == len(et_list) - 1
        ), f"Expected {len(et_list) - 1} interval managers but got {len(interval_managers)}"

        for _interval_manager, (start_time, end_time) in zip(interval_managers, zip(et_list[:-1], et_list[1:])):

            task_kwargs = dict(config["extraction_kwargs"])

            task_kwargs["start_time_isot"] = start_time.isot
            task_kwargs["end_time_isot"] = end_time.isot
            task_kwargs["time_interval_manager_json"] = _interval_manager.to_json()

            task_kwargs["extraction_name"] = name
            task_kwargs["retry_count"] = retry_count

            if not dry_run:
                result = run_data_extraction_task.delay(**task_kwargs)
                logging.info(f"  Task {task_counter}: {start_time.iso} → {end_time.iso} | Task ID: {result.id}")
            else:
                logging.info(f"  Task {task_counter}: {start_time.iso} → {end_time.iso} | (Dry run)")

            task_counter += 1

        logging.info(f"Submitted {task_counter} tasks successfully.")
