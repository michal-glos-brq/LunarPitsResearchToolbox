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

    experiment_name = "lunar_pit_atlas_mapping_LRO_simulation"

    instrument_names = [
        "diviner",
        "lola",
        "mini_rf",
        "lroc_wac",
        "lroc_nac",
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

    filter_type = "point"
    filter_kwargs = {
        "radius_km": 35,
    }
