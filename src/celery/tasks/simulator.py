from app import celery
from astropy.time import Time
import logging

from src.SPICE.kernel_utils.kernel_management import LROKernelManager, GRAILKernelManager
from src.SPICE.instruments import lro as lro_instr
from src.simulators.simulator import RemoteSensingSimulator
from src.simulators.filters import PointFilter
from src.simulators.filters import PointFilter, AreaFilter

logger = logging.getLogger(__name__)



KERNEL_MANAGER_MAP = {
    "LRO": LROKernelManager,
    "GRAIL": GRAILKernelManager,
}

INSTRUMENT_MAP = {
    "diviner": lro_instr.DivinerInstrument,
    "lola": lro_instr.LolaInstrument,
    "mini_rf": lro_instr.MiniRFInstrument,
    "lroc_wac": lro_instr.LROCWACInstrument,
    "lroc_nac": lro_instr.LROCNACInstrument,
}


@celery.task(bind=True)
def run_remote_sensing_simulation(self,
                                  start_time_iso: str,
                                  end_time_iso: str,
                                  kernel_manager_type: str = "LRO",
                                  delete_kernels: bool = False,
                                  instrument_names: list = None) -> dict:
    """
    Celery task to run remote sensing simulation.

    :param start_time_iso: Start time in ISO format (UTC)
    :param end_time_iso: End time in ISO format (UTC)
    :param kernel_manager_type: "LRO" or "GRAIL"
    :param delete_kernels: If True, dynamic kernels will be deleted after use
    :param instrument_names: List of instrument short names (e.g. ["diviner", "lola"])
    :return: Simulation metadata
    """

    try:
        start_time = Time(start_time_iso, scale="utc")
        end_time = Time(end_time_iso, scale="utc")
    except Exception as e:
        raise ValueError(f"Invalid time format: {e}")

    if kernel_manager_type not in KERNEL_MANAGER_MAP:
        raise ValueError(f"Unsupported kernel manager: {kernel_manager_type}")

    if not instrument_names:
        instrument_names = list(INSTRUMENT_MAP.keys())

    try:
        instruments = [INSTRUMENT_MAP[name]() for name in instrument_names]
    except KeyError as e:
        raise ValueError(f"Invalid instrument name: {e}")

    kernel_manager_cls = KERNEL_MANAGER_MAP[kernel_manager_type]
    kernel_manager = kernel_manager_cls(
        frame="MOON_PA_DE440",
        detailed=True,
        pre_download_kernels=True,
        diviner_ck=True,
        lroc_ck=True,
        keep_dynamic_kernels=not delete_kernels,
    )

    kernel_manager.activate()
    filter_obj = PointFilter(35, kernel_manager.static_kernels['dsk'][0].filename)

    logger.info(f"Starting simulation with instruments: {instrument_names}")
    simulator = RemoteSensingSimulator(instruments, filter_obj, kernel_manager)

    simulator.start_simulation(
        start_time=start_time,
        end_time=end_time,
        interactive_progress=False,
        current_task=self,
    )

    logger.info("Simulation completed successfully.")
    return {
        "status": "completed",
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "instruments": instrument_names,
        "kernel_manager": kernel_manager_type,
        "deleted_kernels_after_use": delete_kernels,
    }
