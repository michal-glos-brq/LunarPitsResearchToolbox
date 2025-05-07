from .base_extraction_experiment import BaseExtractionConfig

from astropy.time import Time, TimeDelta


class DIVINERTestExtractorConfig(BaseExtractionConfig):
    """
    Run example: ./src/manual_scripts/assign_tasks.py \
                    --config-name lunar_pit_extraction_test \
                    --name lunar_pit_extraction_test \
                    --task extraction \
                    --dry-run 
    """

    experiment_name = "lunar_pit_extraction_test"

    instrument_names = [
        "DIVINER",
        "LOLA",
        # "MiniRF", Well, this thingy can get stuck on 3 GB + files, which is 60 DIVINER files
        # for just running tasks for testing purposes it's irrelevant
    ]
    # Here we 100 % define the time intervals for data fetching
    interval_name = "lunar_pit_run"
    kernel_manager_type = "LRO"

    start_time = Time("2009-07-05T00:00:00.000", format="isot", scale="utc")
    end_time = start_time + TimeDelta(45, format="jd")
    step_days = 1

    kernel_manager_kwargs = {
        "frame": "MOON_PA_DE440",
        "detailed": True,
        "pre_download_kernels": False,
        "diviner_ck": True,
        "lroc_ck": True,
        "pre_load_static_kernels": True,
        "keep_dynamic_kernels": True,
    }

    filter_type = "lunar_pit"
    filter_kwargs = {
        "hard_radius": 5,  # Km
    }

    custom_filter_kwargs = {
        "MiniRF": {
            "hard_radius": 5,
        }
    }
