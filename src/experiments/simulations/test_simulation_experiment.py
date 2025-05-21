from astropy.time import Time

from .base_simulation_experiment import BaseSimulationConfig


class TestLROShortSimulationConfig(BaseSimulationConfig):
    experiment_name = "test_lro_short_simulation"

    instrument_names = [
        "DIVINER",
        "LOLA",
    ]

    kernel_manager_type = "LRO"

    start_time = Time("2012-07-05T16:50:24.211", format="isot", scale="utc")
    end_time = Time("2012-07-25T16:50:24.211", format="isot", scale="utc")
    step_days = 1  # 1 day step -> 20 tasks over 20 days

    kernel_manager_kwargs = {
        "frame": "MOON_PA_DE440",
        "detailed": True,
        "pre_download_kernels": False,
        "diviner_ck": True,
        "lroc_ck": False,
        "pre_load_static_kernels": True,
        "keep_dynamic_kernels": True,
    }

    filter_type = "lunar_pit"
    filter_kwargs = {
        "hard_radius": 35,
    }
