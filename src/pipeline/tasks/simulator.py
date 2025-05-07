import uuid
import logging
from typing import Optional, Dict

from astropy.time import Time, TimeDelta

from src.SPICE.kernel_utils.kernel_management import LROKernelManager, GRAILKernelManager
from src.SPICE.instruments.lro import (
    DivinerInstrument,
    LolaInstrument,
    MiniRFInstrument,
    LROCWACInstrument,
    LROCNACInstrument,
)
from src.SPICE.instruments.grail import GrailAInstrument, GrailBInstrument
from src.simulation.simulator import RemoteSensingSimulator
from src.filters import FILTER_MAP
from src.global_config import setup_logging

logger = logging.getLogger(__name__)


KERNEL_MANAGER_MAP: Dict[str, type] = {
    "LRO": LROKernelManager,
    "GRAIL": GRAILKernelManager,
}

INSTRUMENT_MAP: Dict[str, type] = {
    DivinerInstrument.name: DivinerInstrument,
    LolaInstrument.name: LolaInstrument,
    MiniRFInstrument.name: MiniRFInstrument,
    LROCWACInstrument.name: LROCWACInstrument,
    LROCNACInstrument.name: LROCNACInstrument,
    GrailAInstrument.name: GrailAInstrument,
    GrailBInstrument.name: GrailBInstrument,
}


# @app.task(bind=True)
def run_remote_sensing_simulation(
    self,
    start_time_isot: str,
    end_time_isot: str,
    instrument_names: list[str],
    kernel_manager_type: str,
    filter_type: str,
    kernel_manager_kwargs: dict,
    filter_kwargs: dict,
    simulation_name: Optional[str] = None,
    retry_count: Optional[int] = None,
    **kwargs,
) -> dict:
    """
    Run a remote sensing simulation with the specified parameters.

    Parameters:
      - start_time_isot (float): Start time (ephemeris time) in seconds.
      - end_time_isot (float): End time (ephemeris time) in seconds.
      - instrument_names (list[str]): List of instrument names to simulate.
      - kernel_manager_type (str): Type of kernel manager ('LRO' or 'GRAIL').
      - filter_type (str): Filter type ('point' or 'area').
      - kernel_manager_kwargs (dict): Additional parameters for kernel manager instantiation.
      - filter_kwargs (dict): Additional parameters for filter instantiation.
      - kwargs: Any extra keyword arguments.

    Returns:
      A summary dictionary of the simulation results.
    """
    # Setup the logging in case TQDM is not supressed, though with celery task, it should be so in every case
    setup_logging()

    logger.info(
        f"Received args: start_time_isot={start_time_isot}, end_time_isot={end_time_isot}, "
        f"instrument_names={instrument_names}, kernel_manager_type={kernel_manager_type}, "
        f"kernel_manager_kwargs={kernel_manager_kwargs}, filter_kwargs={filter_kwargs}, "
        f"filter_type={filter_type}, simulation_name={simulation_name}, "
        f"extra kwargs={kwargs}"
    )

    try:
        start_time = Time(start_time_isot, format="isot", scale="utc")
        end_time = Time(end_time_isot, format="isot", scale="utc")
    except Exception as e:
        raise ValueError(f"Invalid time format: {e}")

    ### Sanity check
    if kernel_manager_type not in KERNEL_MANAGER_MAP:
        raise ValueError(
            f"Unsupported kernel manager: {kernel_manager_type}. Supported types are: {list(KERNEL_MANAGER_MAP.keys())}"
        )

    for instrument in instrument_names:
        if instrument not in INSTRUMENT_MAP:
            raise ValueError(
                f"Unsupported instrument: {instrument}. Supported instruments are: {list(INSTRUMENT_MAP.keys())}"
            )

    ### Merging configurations

    kernel_manager_kwargs.setdefault("min_required_time", start_time - TimeDelta(10, format="sec"))
    kernel_manager_kwargs.setdefault("max_required_time", end_time + TimeDelta(10, format="sec"))

    ### Instantiating and prepare required objects
    kernel_manager = KERNEL_MANAGER_MAP[kernel_manager_type](**kernel_manager_kwargs)
    kernel_manager.activate(start_time)

    filter_obj = FILTER_MAP[filter_type].from_kwargs_and_kernel_manager(kernel_manager, **filter_kwargs)
    instruments = [INSTRUMENT_MAP[name]() for name in instrument_names]

    ### Run the simulation
    simulator = RemoteSensingSimulator(instruments, filter_obj, kernel_manager)

    try:
        simulator.start_simulation(
            start_time=start_time,
            end_time=end_time,
            interactive_progress=False,
            current_task=self,
            supress_error_logs=True,
            simulation_name=simulation_name,
            task_group_id=str(uuid.uuid4()),
            retry_count=retry_count,
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        if self:
            self.update_state(
                state="FAILURE",
                meta={
                    "error": str(e),
                    "exc_type": e.__class__.__name__,
                    "exc_message": str(e),
                    "start_time": start_time.utc.iso,
                    "end_time": end_time.utc.iso,
                },
            )
        raise
    finally:
        kernel_manager.unload_all()

    logger.info("Simulation completed successfully.")
    return {
        "status": "completed",
        "start_time": start_time.utc.iso,
        "end_time": end_time.utc.iso,
    }
