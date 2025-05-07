# src/pipeline/tasks/extraction.py

import uuid
import logging
from typing import List, Optional, Dict

from astropy.time import Time, TimeDelta

from src.SPICE.kernel_utils.kernel_management import LROKernelManager, GRAILKernelManager
from src.filters import FILTER_MAP
from src.SPICE.instruments.lro import (
    DivinerInstrument,
    LolaInstrument,
    MiniRFInstrument,
    LROCWACInstrument,
    LROCNACInstrument,
)
from src.SPICE.instruments.grail import GrailAInstrument, GrailBInstrument
from src.structures import IntervalManager
from src.data_fetchers.data_extractor import DataFetchingEngine
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


# @shared_task(bind=True, name="run_data_extraction")
def run_data_extraction(
    self,
    start_time_isot: str,  # Obtained - Time.iso
    end_time_isot: str,
    instrument_names: List[str],
    kernel_manager_type: str,
    filter_type: str,
    kernel_manager_kwargs: Dict,
    filter_kwargs: Dict,
    time_interval_manager_json: Dict,
    extraction_name: Optional[str] = None,
    retry_count: Optional[int] = None,
    custom_filter_kwargs: Dict = {},
    **kwargs,
) -> Dict:
    """
    Split [start, end) by et_splits and run extraction on each sub‐interval.

    Returns a summary dict with 'status' and number of chunks processed.
    """
    # Setup the logging in case TQDM is not supressed, though with celery task, it should be so in every case
    setup_logging()

    logger.info(
        f"Received args: start_time_isot={start_time_isot}, end_time_isot={end_time_isot}, "
        f"instrument_names={instrument_names}, kernel_manager_type={kernel_manager_type}, "
        f"kernel_manager_kwargs={kernel_manager_kwargs}, filter_kwargs={filter_kwargs}, "
        f"filter_type={filter_type}, extraction_name={extraction_name}; custom_filter_kwargs={custom_filter_kwargs}"
    )

    try:
        start_time = Time(start_time_isot, format="isot", scale="utc")
        end_time = Time(end_time_isot, format="isot", scale="utc")
    except Exception as e:
        raise ValueError(f"Invalid Time inputs: {e}")

    if kernel_manager_type not in KERNEL_MANAGER_MAP:
        raise ValueError(f"Unsupported kernel manager '{kernel_manager_type}'")

    for nm in instrument_names:
        if nm not in INSTRUMENT_MAP:
            raise ValueError(f"Unsupported instrument '{nm}'")

    if filter_type not in FILTER_MAP:
        raise ValueError(f"Unsupported filter type '{filter_type}'")

    kernel_manager_kwargs.setdefault("min_required_time", start_time - TimeDelta(10, format="sec"))
    kernel_manager_kwargs.setdefault("max_required_time", end_time + TimeDelta(10, format="sec"))

    ### Instantiating and prepare required objects
    kernel_manager = KERNEL_MANAGER_MAP[kernel_manager_type](**kernel_manager_kwargs)
    kernel_manager.activate(start_time)

    filter_obj = FILTER_MAP[filter_type].from_kwargs_and_kernel_manager(kernel_manager, **filter_kwargs)
    custom_filter_objects = {
        name: FILTER_MAP[filter_type].from_kwargs_and_kernel_manager(kernel_manager, **kwargs)
        for name, kwargs in custom_filter_kwargs.items()
    }
    instruments = [INSTRUMENT_MAP[name]() for name in instrument_names]

    extractor = DataFetchingEngine(instruments, filter_obj, kernel_manager, custom_filter_objects)

    try:
        extractor.start_extraction(
            IntervalManager.from_json(time_interval_manager_json),
            start_time=start_time,
            end_time=end_time,
            current_task=self,
            interactive_progress=False,
            extraction_name=extraction_name,
            supress_error_logs=True,
            task_group_id=str(uuid.uuid4()),
            retry_count=retry_count,
        )
    except Exception as e:
        import pdb; pdb.set_trace()
        logger.error(f"Error during extraction: {e}")
        if self:
            self.update_state(state="FAILURE", meta={"error": str(e)})
        raise e
    finally:
        kernel_manager.unload_all()
        return {"status": "success"}
