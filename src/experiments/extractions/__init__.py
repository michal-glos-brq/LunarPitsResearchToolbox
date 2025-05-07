from .base_extraction_experiment import BaseExtractionConfig

# It's necessary to import each config in order to use it
from .lunar_pit_data_extraction_test import DIVINERTestExtractorConfig
from .lunar_pit_data_extraction import DIVINERExtractorConfig

__all__ = [
    "BaseExtractionConfig",
]
