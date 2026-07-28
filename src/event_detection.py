"""Backward-compatible facade for the event detection pipeline.

The implementation was split into the `src.event_detection_pipeline` package.
This module re-exports the public API so existing imports keep working.
"""

from src.event_detection_pipeline import (
    DEFAULT_EVENT_END_SECONDS,
    DEFAULT_EVENT_START_SECONDS,
    DEFAULT_HISTORY,
    DatasetSplit,
    DetectionResult,
    EventDetector,
    Scalers,
    SensorGroups,
    ThresholdConfig,
    WaterQualityANN,
    GraspConfig,
    SensorGraspConfig,
    SensorGraspResult,
    aggregate_evaluation_summary,
    build_detector,
    collect_default_data_paths,
    collect_strong_contamination_paths,
    fit_threshold,
    main,
    run_pipeline,
    split_train_test,
    get_chlorine_regression_summary,
    grasp_sensor_subset_search,
)

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
    "GraspConfig",
    "SensorGraspConfig",
    "SensorGraspResult",
    "aggregate_evaluation_summary",
    "get_chlorine_regression_summary",
    "grasp_sensor_subset_search",
    "EventDetector",
    "build_detector",
    "run_pipeline",
    "collect_default_data_paths",
    "collect_strong_contamination_paths",
    "split_train_test",
    "fit_threshold",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
