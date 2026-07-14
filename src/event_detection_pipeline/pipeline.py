from pathlib import Path
from typing import Sequence
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import torch

from .constants import (
    DEFAULT_ARSENIC_THRESHOLD,
    DEFAULT_EVENT_END_SECONDS,
    DEFAULT_EVENT_START_SECONDS,
    DEFAULT_HISTORY,
)
from .data import (
    collect_default_data_paths,
    event_flags_from_time,
    load_npz_file,
    make_supervised_sequences,
    split_file,
    split_train_test,
)
from .model import (
    EventClassificationANN,
    WaterQualityANN,
    apply_scalers,
    fit_scalers,
    inverse_targets,
    predict,
    residual_series,
    select_device,
    train_model,
    train_event_classifier,
    predict_event_probability,
)
from .thresholding import bayesian_event_probability, fit_threshold, rolling_thresholds
from .classes import DatasetSplit, DetectionResult, Scalers, SensorGroups, ThresholdConfig

class EventDetector:
    def __init__(self, models: Sequence[WaterQualityANN], scalers: Sequence[Scalers],
                 thresholds: Sequence[ThresholdConfig], true_positive_rates: Sequence[float],
                 false_positive_rates: Sequence[float], device: torch.device, history: int,
                 groups: SensorGroups, epochs: int = 80, batch_size: int = 128,
                 learning_rate: float = 1e-3, hidden_sizes: Sequence[int] = (128, 64),
                 label_mode: str = "arsenic_arrival",
                 arsenic_threshold: float = DEFAULT_ARSENIC_THRESHOLD,
                 fusion_method: str = "mean", alarm_threshold: float = 0.5,
                 active_target_indices: Sequence[int] | None = None,
                 max_validation_false_alarm_rate: float = 0.1,
                 dropout: float = 0.1, weight_decay: float = 1e-4,
                 early_stopping_patience: int = 12,
                 event_classifier: EventClassificationANN | None = None,
                 classifier_input_mean: np.ndarray | None = None,
                 classifier_input_std: np.ndarray | None = None):
        self.models = list(models)
        self.target_scalers = list(scalers)
        self.thresholds = list(thresholds)
        self.true_positive_rates = np.asarray(true_positive_rates, dtype=np.float32)
        self.false_positive_rates = np.asarray(false_positive_rates, dtype=np.float32)
        self.model = self.models[0]
        self.scalers = self.target_scalers[0]
        self.threshold = self.thresholds[0]
        self.true_positive_rate = float(self.true_positive_rates[0])
        self.false_positive_rate = float(self.false_positive_rates[0])
        self.device = device
        self.history = history
        self.groups = groups
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_sizes = tuple(hidden_sizes)
        self.label_mode = label_mode
        self.arsenic_threshold = arsenic_threshold
        self.fusion_method = fusion_method
        self.alarm_threshold = alarm_threshold
        if active_target_indices is None:
            active_target_indices = range(len(self.models))
        self.active_target_indices = np.asarray(active_target_indices, dtype=int)
        self.max_validation_false_alarm_rate = max_validation_false_alarm_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.event_classifier = event_classifier
        self.classifier_input_mean = classifier_input_mean
        self.classifier_input_std = classifier_input_std

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
        split, _ = split_file(
            path,
            history=self.history,
            event_start_seconds=event_start_seconds,
            event_end_seconds=event_end_seconds,
            label_mode=self.label_mode,
            arsenic_threshold=self.arsenic_threshold,
        )
        predictions = _predict_all_targets(self.models, self.target_scalers, split.inputs, split.targets, self.device)
        residuals = residual_series(predictions, split.targets)
        upper, lower, parameter_probabilities = _score_all_targets(
            residuals,
            self.thresholds,
            self.true_positive_rates,
            self.false_positive_rates,
        )
        probabilities = _fuse_probabilities(
            parameter_probabilities[:, self.active_target_indices],
            method=self.fusion_method,
        )
        if self.event_classifier is not None:
            classifier_inputs = _classification_features(split, self.groups, self.history)
            classifier_inputs = (
                classifier_inputs - self.classifier_input_mean
            ) / self.classifier_input_std
            probabilities = predict_event_probability(
                self.event_classifier, classifier_inputs, self.device
            )
        parameter_alarms = np.column_stack(
            [
                parameter_probabilities[:, index] >= self.thresholds[index].alarm_threshold
                for index in range(parameter_probabilities.shape[1])
            ]
        )
        alarms = probabilities >= self.alarm_threshold

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
            parameter_event_probability=parameter_probabilities,
            parameter_alarms=parameter_alarms,
        )


def _predict_all_targets(
    models: Sequence[WaterQualityANN],
    scalers: Sequence[Scalers],
    inputs: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for target_index, (model, scaler) in enumerate(zip(models, scalers)):
        target_inputs = inputs[:, target_index, :]
        target_values = targets[:, target_index : target_index + 1]
        scaled_inputs, _ = apply_scalers(target_inputs, target_values, scaler)
        scaled_predictions = predict(model, scaled_inputs, device)
        predictions.append(inverse_targets(scaled_predictions, scaler).reshape(-1))
    return np.column_stack(predictions).astype(np.float32)


def _train_all_targets(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    device: torch.device,
    *,
    hidden_sizes: Sequence[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    normal_flags: np.ndarray | None = None,
    validation_inputs: np.ndarray | None = None,
    validation_targets: np.ndarray | None = None,
    validation_flags: np.ndarray | None = None,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 12,
) -> tuple[list[WaterQualityANN], list[Scalers]]:
    models: list[WaterQualityANN] = []
    scalers: list[Scalers] = []

    for target_index in range(train_targets.shape[1]):
        target_inputs = train_inputs[:, target_index, :]
        target_values = train_targets[:, target_index : target_index + 1]
        if normal_flags is not None:
            normal_mask = ~normal_flags[:, target_index]
            target_inputs = target_inputs[normal_mask]
            target_values = target_values[normal_mask]
        scaler = fit_scalers(target_inputs, target_values)
        scaled_inputs, scaled_targets = apply_scalers(target_inputs, target_values, scaler)
        scaled_validation_inputs = None
        scaled_validation_targets = None
        if validation_inputs is not None and validation_targets is not None:
            candidate_inputs = validation_inputs[:, target_index, :]
            candidate_targets = validation_targets[:, target_index : target_index + 1]
            if validation_flags is not None:
                validation_normal_mask = ~validation_flags[:, target_index]
                candidate_inputs = candidate_inputs[validation_normal_mask]
                candidate_targets = candidate_targets[validation_normal_mask]
            scaled_validation_inputs, scaled_validation_targets = apply_scalers(
                candidate_inputs, candidate_targets, scaler
            )
        model = train_model(
            scaled_inputs,
            scaled_targets,
            device,
            hidden_sizes=hidden_sizes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            validation_inputs=scaled_validation_inputs,
            validation_targets=scaled_validation_targets,
            patience=early_stopping_patience,
        )
        models.append(model)
        scalers.append(scaler)

    return models, scalers


def _fit_all_thresholds(
    residuals: np.ndarray,
    event_flags: np.ndarray,
) -> tuple[list[ThresholdConfig], np.ndarray, np.ndarray]:
    thresholds: list[ThresholdConfig] = []
    true_positive_rates: list[float] = []
    false_positive_rates: list[float] = []

    for target_index in range(residuals.shape[1]):
        threshold, true_positive_rate, false_positive_rate, _, _ = fit_threshold(
            residuals[:, target_index],
            event_flags[:, target_index] if event_flags.ndim == 2 else event_flags,
        )
        thresholds.append(threshold)
        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)

    return thresholds, np.asarray(true_positive_rates, dtype=np.float32), np.asarray(false_positive_rates, dtype=np.float32)


def _score_all_targets(
    residuals: np.ndarray,
    thresholds: Sequence[ThresholdConfig],
    true_positive_rates: np.ndarray,
    false_positive_rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper_values: list[np.ndarray] = []
    lower_values: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for target_index, threshold in enumerate(thresholds):
        target_residuals = residuals[:, target_index]
        upper, lower = rolling_thresholds(
            target_residuals,
            threshold.window_fraction,
            threshold.upper_multiplier,
            threshold.lower_multiplier,
            threshold.outlier_upper,
            threshold.outlier_lower,
        )
        target_probabilities = bayesian_event_probability(
            target_residuals,
            upper,
            lower,
            float(true_positive_rates[target_index]),
            float(false_positive_rates[target_index]),
            initial_probability=0.05,
            smoothing=0.6,
        )
        upper_values.append(upper)
        lower_values.append(lower)
        probabilities.append(target_probabilities)

    return np.column_stack(upper_values), np.column_stack(lower_values), np.column_stack(probabilities)


def _fuse_probabilities(
    parameter_probabilities: np.ndarray, method: str = "mean"
) -> np.ndarray:
    if method == "mean":
        return np.mean(parameter_probabilities, axis=1)
    if method == "max":
        return np.max(parameter_probabilities, axis=1)
    if method == "noisy_or":
        return 1.0 - np.prod(1.0 - parameter_probabilities, axis=1)
    raise ValueError(f"Unknown fusion method: {method}")


def _calibrate_alarm_threshold(
    event_flags: np.ndarray,
    event_probability: np.ndarray,
    max_false_alarm_rate: float = 0.1,
) -> float:
    """Maximize validation sensitivity subject to an operational FAR limit."""
    event_flags = np.asarray(event_flags, dtype=bool)
    candidates = np.unique(
        np.quantile(event_probability, np.linspace(0.0, 1.0, 501))
    )
    best_score = (-np.inf, -np.inf)
    best_threshold = 0.5
    for candidate in candidates:
        alarms = event_probability >= candidate
        positives = int(np.sum(event_flags))
        negatives = int(np.sum(~event_flags))
        sensitivity = np.sum(alarms & event_flags) / positives if positives else 0.0
        specificity = np.sum(~alarms & ~event_flags) / negatives if negatives else 0.0
        false_alarm_rate = 1.0 - specificity
        if false_alarm_rate > max_false_alarm_rate:
            continue
        score = (sensitivity, specificity)
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)
    return best_threshold


def _calibrate_balanced_accuracy_threshold(
    event_flags: np.ndarray, event_probability: np.ndarray
) -> float:
    best_score = -np.inf
    best_threshold = 0.5
    for candidate in np.unique(
        np.quantile(event_probability, np.linspace(0.0, 1.0, 1001))
    ):
        alarms = event_probability >= candidate
        sensitivity = np.mean(alarms[event_flags])
        specificity = np.mean(~alarms[~event_flags])
        score = 0.5 * (sensitivity + specificity)
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)
    return best_threshold


def _classification_features(
    split: DatasetSplit, groups: SensorGroups, history: int
) -> np.ndarray:
    """Build current and multiscale temporal features for event classification."""
    flow_count = len(groups.flow)
    flows = split.inputs[:, 0, :flow_count]
    lags = split.inputs[:, :, flow_count : flow_count + history]
    current = split.targets
    last = lags[:, :, -1]
    mean_6 = lags[:, :, -min(6, history) :].mean(axis=2)
    mean_24 = lags[:, :, -min(24, history) :].mean(axis=2)
    mean_history = lags.mean(axis=2)
    cyclical_time = split.inputs[:, 0, -2:]
    return np.column_stack(
        [
            flows, current, last, current - last, current - mean_6,
            current - mean_24, current - mean_history, cyclical_time,
        ]
    ).astype(np.float32)


def _split_train_validation_paths(
    paths: Sequence[Path], validation_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if len(paths) < 2:
        raise ValueError("At least two training scenarios are required for validation")
    rng = np.random.default_rng(seed)
    shuffled = [paths[index] for index in rng.permutation(len(paths))]
    validation_count = max(1, int(round(len(paths) * validation_fraction)))
    validation_count = min(validation_count, len(paths) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def build_detector(
    train_paths: Sequence[Path],
    history: int = DEFAULT_HISTORY,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    hidden_sizes: Sequence[int] = (128, 64),
    event_start_seconds: float = DEFAULT_EVENT_START_SECONDS,
    event_end_seconds: float = DEFAULT_EVENT_END_SECONDS,
    label_mode: str = "arsenic_arrival",
    arsenic_threshold: float = DEFAULT_ARSENIC_THRESHOLD,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    fusion_method: str = "mean",
    max_validation_false_alarm_rate: float = 0.1,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 12,
) -> EventDetector:
    device = select_device()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    fit_paths, validation_paths = _split_train_validation_paths(
        train_paths, validation_fraction, random_seed
    )
    train_splits: list[DatasetSplit] = []
    validation_splits: list[DatasetSplit] = []
    groups: SensorGroups | None = None

    for path in fit_paths:
        split, inferred_groups = split_file(
            path,
            history=history,
            event_start_seconds=event_start_seconds,
            event_end_seconds=event_end_seconds,
            label_mode=label_mode,
            arsenic_threshold=arsenic_threshold,
        )
        groups = inferred_groups
        train_splits.append(split)

    if groups is None:
        raise ValueError("No training files were supplied")

    for path in validation_paths:
        split, inferred_groups = split_file(
            path,
            history=history,
            event_start_seconds=event_start_seconds,
            event_end_seconds=event_end_seconds,
            label_mode=label_mode,
            arsenic_threshold=arsenic_threshold,
        )
        if inferred_groups.chlorine_nodes != groups.chlorine_nodes:
            raise ValueError("Sensor layouts differ between training scenarios")
        validation_splits.append(split)

    train_inputs = np.concatenate([split.inputs for split in train_splits], axis=0)
    train_targets = np.concatenate([split.targets for split in train_splits], axis=0)
    train_target_flags = np.concatenate(
        [split.target_flags for split in train_splits], axis=0
    )
    validation_inputs = np.concatenate(
        [split.inputs for split in validation_splits], axis=0
    )
    validation_targets = np.concatenate(
        [split.targets for split in validation_splits], axis=0
    )
    validation_target_flags = np.concatenate(
        [split.target_flags for split in validation_splits], axis=0
    )
    classifier_train_inputs = np.concatenate(
        [_classification_features(split, groups, history) for split in train_splits]
    )
    classifier_train_flags = np.concatenate(
        [split.flags for split in train_splits]
    )
    classifier_validation_inputs = np.concatenate(
        [_classification_features(split, groups, history) for split in validation_splits]
    )
    classifier_validation_flags = np.concatenate(
        [split.flags for split in validation_splits]
    )
    classifier_input_mean = classifier_train_inputs.mean(axis=0)
    classifier_input_std = classifier_train_inputs.std(axis=0)
    classifier_input_std[classifier_input_std == 0] = 1.0
    scaled_classifier_train_inputs = (
        classifier_train_inputs - classifier_input_mean
    ) / classifier_input_std
    scaled_classifier_validation_inputs = (
        classifier_validation_inputs - classifier_input_mean
    ) / classifier_input_std
    event_classifier = train_event_classifier(
        scaled_classifier_train_inputs,
        classifier_train_flags,
        scaled_classifier_validation_inputs,
        classifier_validation_flags,
        device,
        hidden_sizes=(128, 64),
        learning_rate=3e-4,
        batch_size=256,
        epochs=100,
        dropout=0.1,
        weight_decay=1e-4,
        patience=10,
    )
    models, scalers = _train_all_targets(
        train_inputs,
        train_targets,
        device,
        hidden_sizes=hidden_sizes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        normal_flags=train_target_flags,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        validation_flags=validation_target_flags,
        dropout=dropout,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
    )

    all_event_inputs = np.concatenate([split.inputs for split in validation_splits], axis=0)
    all_event_targets = np.concatenate([split.targets for split in validation_splits], axis=0)
    all_event_flags = np.concatenate(
        [split.target_flags for split in validation_splits], axis=0
    )

    predictions = _predict_all_targets(models, scalers, all_event_inputs, all_event_targets, device)
    residuals = residual_series(predictions, all_event_targets)

    thresholds, true_positive_rates, false_positive_rates = _fit_all_thresholds(residuals, all_event_flags)
    active_target_indices = np.flatnonzero(np.any(all_event_flags, axis=0))
    if active_target_indices.size == 0:
        raise ValueError(
            "No arsenic arrivals exceed the configured threshold in validation"
        )

    detector = EventDetector(
        models=models,
        scalers=scalers,
        thresholds=thresholds,
        true_positive_rates=true_positive_rates,
        false_positive_rates=false_positive_rates,
        device=device,
        history=history,
        groups=groups,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_sizes=hidden_sizes,
        label_mode=label_mode,
        arsenic_threshold=arsenic_threshold,
        fusion_method=fusion_method,
        active_target_indices=active_target_indices,
        max_validation_false_alarm_rate=max_validation_false_alarm_rate,
        dropout=dropout,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        event_classifier=event_classifier,
        classifier_input_mean=classifier_input_mean,
        classifier_input_std=classifier_input_std,
    )
    validation_results = [
        detector.detect(
            path,
            event_start_seconds=event_start_seconds,
            event_end_seconds=event_end_seconds,
        )
        for path in validation_paths
    ]
    detector.alarm_threshold = _calibrate_balanced_accuracy_threshold(
        np.concatenate([result.event_flags for result in validation_results]),
        np.concatenate([result.event_probability for result in validation_results]),
    )
    return detector


def run_pipeline(
    data_dir: Path,
    history: int = DEFAULT_HISTORY,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    hidden_sizes: Sequence[int] = (128, 64),
    event_start_seconds: float = DEFAULT_EVENT_START_SECONDS,
    event_end_seconds: float = DEFAULT_EVENT_END_SECONDS,
    label_mode: str = "arsenic_arrival",
    arsenic_threshold: float = DEFAULT_ARSENIC_THRESHOLD,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    fusion_method: str = "mean",
    max_validation_false_alarm_rate: float = 0.1,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 12,
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
        label_mode=label_mode,
        arsenic_threshold=arsenic_threshold,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
        fusion_method=fusion_method,
        max_validation_false_alarm_rate=max_validation_false_alarm_rate,
        dropout=dropout,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
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
            "window_fraction": [threshold.window_fraction for threshold in detector.thresholds],
            "upper_multiplier": [threshold.upper_multiplier for threshold in detector.thresholds],
            "lower_multiplier": [threshold.lower_multiplier for threshold in detector.thresholds],
            "alarm_threshold": [threshold.alarm_threshold for threshold in detector.thresholds],
        },
        "calibration": {
            "fusion_method": detector.fusion_method,
            "alarm_threshold": detector.alarm_threshold,
            "label_mode": detector.label_mode,
            "arsenic_threshold": detector.arsenic_threshold,
            "max_validation_false_alarm_rate": detector.max_validation_false_alarm_rate,
            "active_target_nodes": [
                detector.groups.chlorine_nodes[index]
                for index in detector.active_target_indices
            ],
        },
        "train_files": [path.name for path in train_paths],
        "test_files": [path.name for path in test_paths],
    }
