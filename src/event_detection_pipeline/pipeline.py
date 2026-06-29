from pathlib import Path
from typing import Sequence
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import torch

from .constants import DEFAULT_EVENT_END_SECONDS, DEFAULT_EVENT_START_SECONDS, DEFAULT_HISTORY
from .data import (
    collect_default_data_paths,
    event_flags_from_time,
    load_npz_file,
    make_supervised_sequences,
    split_file,
    split_train_test,
)
from .model import (
    WaterQualityANN,
    apply_scalers,
    fit_scalers,
    inverse_targets,
    predict,
    residual_series,
    select_device,
    train_model,
)
from .thresholding import bayesian_event_probability, fit_threshold, rolling_thresholds
from .classes import DatasetSplit, DetectionResult, Scalers, SensorGroups, ThresholdConfig

class EventDetector:
    def __init__(self, model: WaterQualityANN, scalers: Scalers, 
                 threshold: ThresholdConfig, true_positive_rate: float, 
                 false_positive_rate: float, device: torch.device, history: int, 
                 groups: SensorGroups):
        self.model = model
        self.scalers = scalers
        self.threshold = threshold
        self.true_positive_rate = true_positive_rate
        self.false_positive_rate = false_positive_rate
        self.device = device
        self.history = history
        self.groups = groups

    def prediction_confusion_matrix( self, path: Path, event_start_seconds: 
        float = DEFAULT_EVENT_START_SECONDS, event_end_seconds: 
        float = DEFAULT_EVENT_END_SECONDS,):
        result = self.detect(
            path,
            event_start_seconds=event_start_seconds,
            event_end_seconds=event_end_seconds,
        )

        cm = confusion_matrix(
            result.event_flags,
            result.alarms,
            labels=[False, True],
            normalize='all' 
        )
        disp = ConfusionMatrixDisplay(cm)
        return disp

    def detect(self, path: Path, *, 
               event_start_seconds: float = DEFAULT_EVENT_START_SECONDS, 
               event_end_seconds: float = DEFAULT_EVENT_END_SECONDS) -> DetectionResult:
        loaded = load_npz_file(path)
        flags = event_flags_from_time(loaded["sensor_readings_time"], event_start_seconds, event_end_seconds)
        split = make_supervised_sequences(loaded["sensor_readings"], self.groups, self.history, loaded["sensor_readings_time"], flags)
        scaled_inputs, _ = apply_scalers(split.inputs, split.targets, self.scalers)
        scaled_predictions = predict(self.model, scaled_inputs, self.device)
        predictions = inverse_targets(scaled_predictions, self.scalers)

        residuals = residual_series(predictions, split.targets)
        upper, lower = rolling_thresholds(
            residuals,
            self.threshold.window_fraction,
            self.threshold.upper_multiplier,
            self.threshold.lower_multiplier,
            self.threshold.outlier_upper,
            self.threshold.outlier_lower,
        )
        probabilities = bayesian_event_probability(
            residuals,
            upper,
            lower,
            self.true_positive_rate,
            self.false_positive_rate,
            initial_probability=1e-5,
            smoothing=0.6,
        )
        alarms = probabilities >= self.threshold.alarm_threshold

        return DetectionResult(
            times=split.times,
            residuals=residuals,
            upper_threshold=upper,
            lower_threshold=lower,
            event_probability=probabilities,
            alarms=alarms,
            event_flags=split.flags,
            predicted=predictions,
            actual=split.targets,
        )


def build_detector(
    train_paths: Sequence[Path],
    history: int = DEFAULT_HISTORY,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    hidden_sizes: Sequence[int] = (128, 64),
    event_start_seconds: float = DEFAULT_EVENT_START_SECONDS,
    event_end_seconds: float = DEFAULT_EVENT_END_SECONDS,
) -> EventDetector:
    device = select_device()
    train_splits: list[DatasetSplit] = []
    groups: SensorGroups | None = None

    for path in train_paths:
        split, inferred_groups = split_file(
            path,
            history=history,
            event_start_seconds=event_start_seconds,
            event_end_seconds=event_end_seconds,
        )
        groups = inferred_groups
        train_splits.append(split)

    if groups is None:
        raise ValueError("No training files were supplied")

    normal_inputs: list[np.ndarray] = []
    normal_targets: list[np.ndarray] = []
    event_inputs: list[np.ndarray] = []
    event_targets: list[np.ndarray] = []
    event_flags: list[np.ndarray] = []

    for split in train_splits:
        # inverting the flags with `~`
        normal_mask = ~split.flags
        normal_inputs.append(split.inputs[normal_mask])
        normal_targets.append(split.targets[normal_mask])
        event_inputs.append(split.inputs)
        event_targets.append(split.targets)
        event_flags.append(split.flags)

    train_inputs = np.concatenate(normal_inputs, axis=0)
    train_targets = np.concatenate(normal_targets, axis=0)
    scalers = fit_scalers(train_inputs, train_targets)
    scaled_train_inputs, scaled_train_targets = apply_scalers(train_inputs, train_targets, scalers)
    model = train_model(
        scaled_train_inputs,
        scaled_train_targets,
        device,
        hidden_sizes=hidden_sizes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    all_event_inputs = np.concatenate(event_inputs, axis=0)
    all_event_targets = np.concatenate(event_targets, axis=0)
    all_event_flags = np.concatenate(event_flags, axis=0)

    scaled_event_inputs, _ = apply_scalers(all_event_inputs, all_event_targets, scalers)
    scaled_predictions = predict(model, scaled_event_inputs, device)
    predictions = inverse_targets(scaled_predictions, scalers)
    residuals = residual_series(predictions, all_event_targets)

    threshold, true_positive_rate, false_positive_rate, _, _ = fit_threshold(residuals, all_event_flags)

    return EventDetector(
        model=model,
        scalers=scalers,
        threshold=threshold,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
        device=device,
        history=history,
        groups=groups,
    )


def run_pipeline(
    data_dir: Path,
    history: int = DEFAULT_HISTORY,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    hidden_sizes: Sequence[int] = (128, 64),
    event_start_seconds: float = DEFAULT_EVENT_START_SECONDS,
    event_end_seconds: float = DEFAULT_EVENT_END_SECONDS,
) -> dict[str, object]:
    paths = collect_default_data_paths(data_dir)
    train_paths, test_paths = split_train_test(paths)
    detector = build_detector(
        train_paths,
        history=history,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_sizes=hidden_sizes,
        event_start_seconds=event_start_seconds,
        event_end_seconds=event_end_seconds,
    )

    train_results = [
        detector.detect(path, event_start_seconds=event_start_seconds, event_end_seconds=event_end_seconds)
        for path in train_paths
    ]
    test_results = [
        detector.detect(path, event_start_seconds=event_start_seconds, event_end_seconds=event_end_seconds)
        for path in test_paths
    ]

    def aggregate(results: Sequence[DetectionResult]) -> dict[str, float]:
        if not results:
            return {
                "precision": 0.0,
                "accuracy": 0.0,
                "probability_of_detection": 0.0,
                "false_alarm_rate": 0.0,
                "detection_latency_seconds": 0.0,
                "alarms": 0.0,
            }
        summaries = [result.summary() for result in results]
        return {
            "precision": float(np.mean([summary["precision"] for summary in summaries])),
            "accuracy": float(np.mean([summary["accuracy"] for summary in summaries])),
            "probability_of_detection": float(np.mean([summary.get("probability_of_detection", summary.get("recall", 0.0)) for summary in summaries])),
            "false_alarm_rate": float(np.mean([summary["false_alarm_rate"] for summary in summaries])),
            "detection_latency_seconds": float(np.nanmean([summary.get("detection_latency_seconds", np.nan) for summary in summaries])),
            "alarms": float(np.sum([summary["alarms"] for summary in summaries])),
        }

    return {
        "device": str(detector.device),
        "train": aggregate(train_results),
        "test": aggregate(test_results),
        "threshold": {
            "window_fraction": detector.threshold.window_fraction,
            "upper_multiplier": detector.threshold.upper_multiplier,
            "lower_multiplier": detector.threshold.lower_multiplier,
            "alarm_threshold": detector.threshold.alarm_threshold,
        },
        "train_files": [path.name for path in train_paths],
        "test_files": [path.name for path in test_paths],
    }
