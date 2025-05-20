class BaseExtractionConfig:
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
    # Is used when custom filter is needed for particular instrument for whatever reason
    custom_filter_kwargs = {}
    interval_name = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls.experiment_name] = cls

    @staticmethod
    def get_config_dict(config_name: str):
        if config_name not in BaseExtractionConfig.registry:
            available = list(BaseExtractionConfig.registry.keys())
            raise ValueError(f"Unknown experiment config '{config_name}'. Available configs: {available}")

        config_class = BaseExtractionConfig.registry[config_name]
        return config_class.to_dict()

    @classmethod
    def to_dict(cls):
        return {
            "extraction_kwargs": {
                "instrument_names": cls.instrument_names,
                "kernel_manager_type": cls.kernel_manager_type,
                # When less than 300-400 GB of free space, set to False
                "keep_dynamic_kernels": cls.kernel_manager_kwargs.get("keep_dynamic_kernels", False),
                "filter_type": cls.filter_type,
                "simulation_name": cls.experiment_name,
                "kernel_manager_kwargs": cls.kernel_manager_kwargs,
                "filter_kwargs": cls.filter_kwargs,
                "custom_filter_kwargs": cls.custom_filter_kwargs,
            },
            # Collecting intervals with this name
            "interval_name": cls.interval_name,
            "experiment_name": cls.experiment_name,
            "start_time": cls.start_time,
            "end_time": cls.end_time,
            "step_days": cls.step_days,
        }
