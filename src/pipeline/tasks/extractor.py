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
    logger.info(f"Starting extraction {extraction_name or '<unnamed>'}, retry={retry_count}")

    # 1. Parse & validate times up-front
    try:
        start_time = Time(start_time_isot, format="isot", scale="utc")
        end_time   = Time(end_time_isot,   format="isot", scale="utc")
    except Exception as e:
        raise ValueError(f"Bad time input: {e!r}")

    if start_time >= end_time:
        raise ValueError("start_time must be strictly before end_time")

    # 2. Validate parameters
    if kernel_manager_type not in KERNEL_MANAGER_MAP:
        raise ValueError(f"Unknown kernel_manager_type: {kernel_manager_type!r}")
    for name in instrument_names:
        if name not in INSTRUMENT_MAP:
            raise ValueError(f"Unknown instrument: {name!r}")
    if filter_type not in FILTER_MAP:
        raise ValueError(f"Unknown filter_type: {filter_type!r}")

    # 3. Tweak kernel bounds by a small margin
    kernel_manager_kwargs.setdefault("min_required_time", start_time - TimeDelta(10, format="sec"))
    kernel_manager_kwargs.setdefault("max_required_time", end_time + TimeDelta(10, format="sec"))

    # 4. Instantiate everything inside a context so we guarantee cleanup
    km_cls = KERNEL_MANAGER_MAP[kernel_manager_type]
    kernel_manager = km_cls(**kernel_manager_kwargs)
    kernel_manager.activate(start_time)

    try:
        filter_obj = FILTER_MAP[filter_type].from_kwargs_and_kernel_manager(kernel_manager, **filter_kwargs)
        custom_filter_objects = {
            name: FILTER_MAP[filter_type].from_kwargs_and_kernel_manager(kernel_manager, **kwargs)
            for name, kwargs in custom_filter_kwargs.items()
        }
        instruments = [INSTRUMENT_MAP[name]() for name in instrument_names]

        # 6. Kick off the engine
        extractor = DataFetchingEngine(instruments, filter_obj, kernel_manager, custom_filter_objects)  
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
        # 7. Capture full exception info in Celery meta so exc_type isn’t lost
        import traceback, sys
        etype, evalue, tb = sys.exc_info()
        tb_lines = traceback.format_exception(etype, evalue, tb)
        logger.error("Extraction failed:\n%s", "".join(tb_lines))
        if self:
            self.update_state(
                state="FAILURE",
                meta={
                    "exc_type":    etype.__name__,
                    "exc_message": str(evalue),
                    "traceback":   tb_lines,
                }
            )
        raise  # re-raise with original traceback

    finally:
        # 8. Always unload kernels no matter what
        kernel_manager.unload_all()

    # 9. Return a clear summary
    return {
        "status":            "SUCCESS",
        "instruments":       instrument_names,
        "retry_count":       retry_count,
        "extraction_name":   extraction_name,
    }