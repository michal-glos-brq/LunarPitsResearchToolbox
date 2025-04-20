import logging

from astropy.time import Time
from tqdm import tqdm
import spiceypy as spice

from src.global_config import SPICE_DECIMAL_PRECISION

def et2astropy_time(et: float) -> Time:
    return Time(spice.et2utc(et, "ISOC", SPICE_DECIMAL_PRECISION), format="isot", scale="utc")


class SPICELog:

    interactive_progress: bool = True
    supress_output: bool = False

    @staticmethod
    def log_spice_exception(e: Exception, context: str = ""):
        """
        Log the exception with its type and any SPICE error state.
        """
        try:
            if SPICELog.supress_output:
                return
            err_type = type(e).__name__
            msg = f"{context} Exception: [{err_type}] {e}"
            if spice.failed():
                spice_error_message = spice.getmsg("SHORT")
                msg += f" | SPICE error: {spice_error_message}"
                spice.reset()
            if SPICELog.interactive_progress:
                tqdm.write(msg)
            else:
                logging.warning(msg)
        except Exception:
            ...