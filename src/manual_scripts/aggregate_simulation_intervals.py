#! /usr/bin/env python3
"""
This is pretty simple script, which only aggregates simulation data into intervals
"""

import logging
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

from src.db.interface import Sessions
from src.experiments.simulations.lunar_pit_simulation import BaseSimulationConfig
from src.global_config import TQDM_NCOLS
from src.global_config import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


def merge_intervals(
    timestamps: List[float], base_step: float, margin: float = 0.1, correction: float = 1.6
) -> List[Tuple[float, float]]:
    """
    Merge a sorted list of timestamps (ephemeris time as float) into continuous intervals.

    Two consecutive timestamps are merged if the gap between them is less than (base_step + margin).

    Parameters:
      timestamps (list[float]): A sorted list of timestamps (ET in seconds).
      base_step (float): The base step threshold (in seconds) from the simulation metadata.
      margin (float): Additional tolerance (in seconds).
      correction (float): Because of sampling, half of the period is added to the interval from both sides (default 1.6 is reasonable for low lunar orbit - inclusive).

    Returns:
      List of tuples, where each tuple is (start_et, end_et) representing a merged continuous interval.
    """
    if not timestamps:
        return []

    # Already sorted by caller—or sort here for extra safety.
    timestamps.sort()
    intervals = []
    current_start = timestamps[0]
    previous = timestamps[0]
    threshold = base_step + margin

    for t in timestamps[1:]:
        if t - previous > threshold:
            intervals.append((current_start - correction, previous + correction))
            current_start = t
        previous = t

    # Append last interval.
    intervals.append((current_start - correction, previous + correction))

    merged_corrected_interval = [intervals.pop(0)]
    for start, end in intervals:
        if start <= merged_corrected_interval[-1][1]:
            merged_corrected_interval[-1] = (
                merged_corrected_interval[-1][0],
                max(merged_corrected_interval[-1][1], end),
            )
        else:
            merged_corrected_interval.append((start, end))

    return intervals


def aggregate_simulation_intervals(
    config_name: str,
    simulation_names: List[str],
    threshold: Optional[float] = None,
    requested_instruments: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Aggregates simulation point timestamps into merged time intervals for each instrument.

    Workflow:
      1. Load simulation configuration settings via BaseSimulationConfig.
      2. Construct a filter name (or ID) from the filter mapping.
      3. Query the simulation metadata documents using Sessions.simulation_tasks_query().
         (These documents should include at least the simulation "_id", "start_time", and "base_step".)
      4. Sort the simulation tasks by their start time.
      5. Obtain the "base_step" from (for example) the first simulation task.
      6. For each instrument:
           - Retrieve its simulation point collection via Sessions.prepare_simulation_collections().
           - For each simulation task, query all documents where `"meta.simulation_id"` matches the task’s `_id`
             (only returning the `"et"` field).
           - Merge all collected ET values into continuous intervals using the merge_intervals() helper.
      7. Return a dictionary mapping each instrument name to its list of merged intervals.

    Parameters:
      config_name (str): Name of the simulation configuration to use.
      simulation_names (List[str]): Simulation name identifiers (should match the one used in the metadata).

    Returns:
      A dict mapping instrument names (str) to lists of (start_et, end_et) tuples.
    """
    # Load the simulation configuration. The configuration is expected to be a dictionary.
    config = BaseSimulationConfig.get_config_dict(config_name)
    instrument_names = config["simulation_kwargs"]["instrument_names"]

    if requested_instruments and not all(
        [req_instrument in instrument_names for req_instrument in requested_instruments]
    ):
        raise ValueError(
            f"Requested instruments {requested_instruments} are not part of the simulation configuration {config_name}."
        )

    if requested_instruments is not None:
        instrument_names = [
            instrument_name for instrument_name in instrument_names if instrument_name in requested_instruments
        ]

    # Query simulation metadata documents. (Assuming this method is implemented accordingly.)
    simulation_tasks = Sessions.simulation_tasks_query(
        simulation_names=simulation_names,
        instrument_names=instrument_names,
    )

    if not simulation_tasks:
        logger.warning("No simulation metadata documents were found for simulation '%s'.", ",".join(simulation_names))
        return {}

    # Sort simulation metadata documents by a time field (e.g., "start_time").
    simulation_tasks.sort(key=lambda doc: doc["start_time"])

    # Assume that all simulation tasks for this simulation share the same base step.
    base_step = simulation_tasks[0].get("base_step")
    if base_step is None:
        raise ValueError("Missing 'base_step' in the simulation metadata documents.")
    base_step = float(base_step)

    # Obtain positive collections for each instrument.
    instrument_collections = {
        instrument: Sessions.prepare_simulation_collections(instrument)[0] for instrument in instrument_names
    }

    intervals_per_instrument = defaultdict(list)

    # For each instrument, gather simulation point timestamps.
    for instrument in instrument_names:
        logger.info("Processing instrument: %s", instrument)
        collection = instrument_collections[instrument]
        all_et_values = []

        # Iterate over each simulation metadata document.
        for task_doc in tqdm(simulation_tasks, desc=f"Iterating through simulations: {instrument}", ncols=TQDM_NCOLS):
            sim_id = task_doc["_id"]
            # Query the simulation points for this simulation task, sorting by "et" in ascending order. Tiemout to 100 minutes
            query = {"meta.simulation_id": sim_id}
            if threshold is not None:
                query["bound_distance"] = {"$lte": threshold}
            cursor = collection.find(query, {"et": 1}, max_time_ms=6000000).sort("et", 1)
            for doc in cursor:
                if "et" in doc:
                    all_et_values.append(doc["et"])

        if not all_et_values:
            logger.warning("No simulation points found for instrument '%s'.", instrument)
            continue

        # Merge the ET values into intervals.
        merged_intervals = merge_intervals(all_et_values, base_step, margin=0.1)
        intervals_per_instrument[instrument] = merged_intervals
        logger.info("Instrument '%s': %d intervals found.", instrument, len(merged_intervals))

    return dict(intervals_per_instrument)


if __name__ == "__main__":

    from src.global_config import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Merge simulation timestamps into time intervals.")
    parser.add_argument("--config-name", help="Name of the experiment config to use.", required=True)
    parser.add_argument("--sim-names", help="Names of the simulation runs to use, delimited by ','.", required=True)
    parser.add_argument("--instruments", help="Names of instruments to use, delimited by ','.", required=False)
    parser.add_argument(
        "--interval-name",
        help="Name of the intervals to save for future extraction. This will be referrenced when assigning extraction tasks.",
        required=True,
    )
    parser.add_argument(
        "--threshold",
        help="Treshold for additional filtering of data (half FOV widths implicitly added).",
        required=False,
        type=float,
        default=None,
    )
    args = parser.parse_args()

    sim_names = list({name.strip() for name in args.sim_names.split(",")})
    result = aggregate_simulation_intervals(
        args.config_name, sim_names, threshold=args.threshold, requested_instruments=args.instruments.split(",")
    )
    Sessions.insert_simulation_intervals(sim_names, result, args.threshold, args.interval_name)
