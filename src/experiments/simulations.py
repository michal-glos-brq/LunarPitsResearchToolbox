"""
Here we store all configurations for simulations of satellites and their instruments - to obtain time intervals of flyovers
and instrument views.
"""


from astropy.time import Time


class BaseSimulationConfig:
    registry = {}

    experiment_name = None
    instrument_names = []
    kernel_manager_type = None
    start_time = None
    end_time = None
    step_days = 1

    kernel_manager_kwargs = {}
    filter_type = None
    filter_kwargs = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.experiment_name] = cls

    @staticmethod
    def get_config_dict(config_name: str):
        if config_name not in BaseSimulationConfig.registry:
            available = list(BaseSimulationConfig.registry.keys())
            raise ValueError(f"Unknown experiment config '{config_name}'. Available configs: {available}")

        config_class = BaseSimulationConfig.registry[config_name]
        return config_class.to_dict()

    @classmethod
    def to_dict(cls):
        return {
            "simulation_kwargs": {
                "instrument_names": cls.instrument_names,
                "kernel_manager_type": cls.kernel_manager_type,
                # When less than 300-400 GB of free space, set to False
                "keep_dynamic_kernels": cls.kernel_manager_kwargs.get("keep_dynamic_kernels", False),
                "filter_type": cls.filter_type,
                "simulation_name": cls.experiment_name,
                "kernel_manager_kwargs": cls.kernel_manager_kwargs,
                "filter_kwargs": cls.filter_kwargs,
            },
            "start_time": cls.start_time,
            "end_time": cls.end_time,
            "step_days": cls.step_days,
        }


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
                            --interval-name test_full_run \
                            --threshold 5
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

class TestLROShortSimulationConfig(BaseSimulationConfig):
    """
    Run example: ./src/manual_scripts/assign_tasks.py --config-name test_lro_short_simulation --name demo_run --dry-run --task remote_sensing
    """
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


