from .base_extraction_experiment import BaseExtractionConfig

from astropy.time import Time



class DIVINERTestExtractorConfig(BaseExtractionConfig):
    """
    Run example: ./src/manual_scripts/assign_tasks.py \
                    --config-name lunar_pit_extraction \
                    --name lunar_pit_extraction \
                    --task extraction \
                    --dry-run 
    """


    experiment_name = "lunar_pit_extraction"

    instrument_names = [
        "DIVINER",
        "LOLA",
        "MiniRF",
    ]
    # Here we 100 % define the time intervals for data fetching
    interval_name = "lunar_pit_run"
    kernel_manager_type = "LRO"

    start_time = Time("2009-07-05T00:00:00.000", format="isot", scale="utc")
    end_time = Time("2024-12-15T00:00:00.000", format="isot", scale="utc")
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
        "hard_radius": 5, # Km
    }

    custom_filter_kwargs = {
        "MiniRF": {
            "hard_radius": 10,
        }
    }

