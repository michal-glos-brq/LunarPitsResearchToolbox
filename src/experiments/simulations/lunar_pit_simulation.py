from astropy.time import Time

from .base_simulation_experiment import BaseSimulationConfig


class LunarPitAtlasMappingLROConfig(BaseSimulationConfig):
    """
    Run example: ./src/manual_scripts/assign_tasks.py \
                        --config-name lunar_pit_atlas_mapping_LRO_simulation \
                        --name lunar_pit_run \
                        --task remote_sensing \
                        --dry-run 
    Run to aggregate: ./src/manual_scripts/aggregate_simulation_intervals.py \
                            --config-name lunar_pit_atlas_mapping_LRO_simulation \
                            --sim-name lunar_pit_run \
                            --instruments DIVINER,LOLA \
                            --interval-name lunar_pit_run \
                            --threshold 5
    Run to aggregate (radar data): ./src/manual_scripts/aggregate_simulation_intervals.py \
                            --config-name lunar_pit_atlas_mapping_LRO_simulation \
                            --sim-name lunar_pit_run \
                            --instruments MiniRF \
                            --interval-name lunar_pit_run \
                            --threshold 20
    """
    experiment_name = "lunar_pit_atlas_mapping_LRO_simulation"

    instrument_names = [
        "DIVINER",
        "LOLA",
        "MiniRF",
        "LROC_WAC",
        "LROC_NAC",
    ]

    kernel_manager_type = "LRO"

    start_time = Time("2009-07-05T16:50:24.211", format="isot", scale="utc")
    end_time = Time("2024-12-15T00:00:00.000", format="isot", scale="utc")
    step_days = 14

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
    # This is used for simulation, not necessirily very exclusive
    filter_kwargs = {
        "hard_radius": 35,
    }


