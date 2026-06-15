from .cli import main
from .constants import DEFAULT_EVENT_END_SECONDS, DEFAULT_EVENT_START_SECONDS, DEFAULT_HISTORY
from .data import collect_default_data_paths, split_train_test
from .model import WaterQualityANN
from .pipeline import EventDetector, build_detector, run_pipeline
from .thresholding import fit_threshold
from .classes import DatasetSplit, DetectionResult, Scalers, SensorGroups, ThresholdConfig

__all__ = [
    "DEFAULT_EVENT_END_SECONDS",
    "DEFAULT_EVENT_START_SECONDS",
    "DEFAULT_HISTORY",
    "SensorGroups",
    "DatasetSplit",
    "Scalers",
    "ThresholdConfig",
    "DetectionResult",
    "WaterQualityANN",
    "EventDetector",
    "build_detector",
    "run_pipeline",
    "collect_default_data_paths",
    "split_train_test",
    "fit_threshold",
    "main",
]
