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
                # "keep_dynamic_kernels": cls.kernel_manager_kwargs.get("keep_dynamic_kernels", False),
                "filter_type": cls.filter_type,
                "simulation_name": cls.experiment_name,
                "kernel_manager_kwargs": cls.kernel_manager_kwargs,
                "filter_kwargs": cls.filter_kwargs,
            },
            "start_time": cls.start_time,
            "end_time": cls.end_time,
            "step_days": cls.step_days,
        }
