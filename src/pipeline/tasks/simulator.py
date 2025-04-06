from astropy.time import Time, TimeDelta
import logging
from typing import Optional

from src.pipeline.app import app
from src.SPICE.kernel_utils.kernel_management import LROKernelManager, GRAILKernelManager
from src.SPICE.instruments import lro as lro_instruments
from src.SPICE.instruments import grail as grail_instruments
from src.simulation.simulator import RemoteSensingSimulator
from src.simulation.filters import PointFilter, AreaFilter


logger = logging.getLogger(__name__)


KERNEL_MANAGER_MAP = {
    "LRO": LROKernelManager,
    "GRAIL": GRAILKernelManager,
}

# We can lay them out like this, because sanity checks are done further down the line
# Simulation e.b. works with only one satellite (is checked)
INSTRUMENT_MAP = {
    "diviner": lro_instruments.DivinerInstrument,
    "lola": lro_instruments.LolaInstrument,
    "mini_rf": lro_instruments.MiniRFInstrument,
    "lroc_wac": lro_instruments.LROCWACInstrument,
    "lroc_nac": lro_instruments.LROCNACInstrument,
    "grai_a": grail_instruments.GrailAInstrument,
    "grai_b": grail_instruments.GrailBInstrument,
}

FILTER_MAP = {
    "point": PointFilter,
    "area": AreaFilter,
}


@app.task(bind=True)
def run_remote_sensing_simulation(
    self,
    start_time_et: float,
    end_time_et: float,
    instrument_names: list[str],
    kernel_manager_type: str,
    filter_type: str,
    kernel_manager_kwargs: dict,
    filter_kwargs: dict,
    simulation_name: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Run a remote sensing simulation with the specified parameters.

    Parameters:
      - start_time_et (float): Start time (ephemeris time) in seconds.
      - end_time_et (float): End time (ephemeris time) in seconds.
      - instrument_names (list[str]): List of instrument names to simulate.
      - kernel_manager_type (str): Type of kernel manager ('LRO' or 'GRAIL').
      - filter_type (str): Filter type ('point' or 'area').
      - kernel_manager_kwargs (dict): Additional parameters for kernel manager instantiation.
      - filter_kwargs (dict): Additional parameters for filter instantiation.
      - kwargs: Any extra keyword arguments.

    Returns:
      A summary dictionary of the simulation results.
    """

    logger.info(
        f"Received args: start_time_et={start_time_et}, end_time_et={end_time_et}, "
        f"instrument_names={instrument_names}, kernel_manager_type={kernel_manager_type}, "
        f"kernel_manager_kwargs={kernel_manager_kwargs}, filter_kwargs={filter_kwargs}, "
        f"filter_type={filter_type}, simulation_name={simulation_name}, "
        f"extra kwargs={kwargs}"
    )

    try:
        # Astropy works for 8 decimal places, hence the 8
        start_time = Time(start_time_et, format="cxcsec", scale="tdb")
        end_time = Time(end_time_et, format="cxcsec", scale="tdb")
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
    kernel_manager_kwargs.setdefault("simulation_name", simulation_name)

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
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "start_time": start_time.utc.iso,
            "end_time": end_time.utc.iso,
        }
    finally:
        kernel_manager.unload_all()

    logger.info("Simulation completed successfully.")
    return {
        "status": "completed",
        "start_time": start_time.utc.iso,
        "end_time": end_time.utc.iso,
    }
