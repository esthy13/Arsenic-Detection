import numpy as np

class SensorGroups:
    def __init__(self, chlorine: list[int], flow: list[int], 
                 arsenic: list[int], chlorine_nodes: list[str] | None = None,
                 arsenic_nodes: list[str] | None = None):
        self.chlorine = chlorine
        self.flow = flow
        self.arsenic = arsenic
        self.chlorine_nodes = chlorine_nodes or [str(index) for index in chlorine]
        self.arsenic_nodes = arsenic_nodes or [str(index) for index in arsenic]

class DatasetSplit:
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, 
                 flags: np.ndarray, times: np.ndarray,
                 target_flags: np.ndarray | None = None):
        self.inputs = inputs
        self.targets = targets
        self.flags = flags
        self.times = times
        if target_flags is None:
            target_flags = np.repeat(flags[:, None], targets.shape[1], axis=1)
        self.target_flags = np.asarray(target_flags, dtype=bool)

class Scalers:
    def __init__(self, input_mean: np.ndarray, input_std: np.ndarray, 
                 target_mean: np.ndarray, target_std: np.ndarray):
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

class ThresholdConfig:
    def __init__(self, window_size_seconds: float, window_fraction: float, 
                 upper_multiplier: float, lower_multiplier: float, 
                 outlier_upper: float, outlier_lower: float, 
                 alarm_threshold = 0.7):
        self.window_size_seconds = window_size_seconds
        self.window_fraction = window_fraction
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self.outlier_upper = outlier_upper
        self.outlier_lower = outlier_lower
        self.alarm_threshold = alarm_threshold


class DetectionResult:
    def __init__(self, times: np.ndarray, residuals: np.ndarray, 
                 upper_threshold: np.ndarray, lower_threshold: np.ndarray, 
                 event_probability: np.ndarray, alarms: np.ndarray, 
                 event_flags: np.ndarray, predicted: np.ndarray, 
                 actual: np.ndarray, parameter_event_probability: np.ndarray | None = None,
                 parameter_alarms: np.ndarray | None = None):
         self.times = times
         self.residuals = residuals
         self.upper_threshold = upper_threshold
         self.lower_threshold = lower_threshold
         self.event_probability = event_probability
         self.alarms = alarms
         self.event_flags = event_flags
         self.predicted = predicted
         self.actual = actual
         self.parameter_event_probability = parameter_event_probability
         self.parameter_alarms = parameter_alarms

    def _event_start_time(self) -> float:
        event_mask = np.asarray(self.event_flags, dtype=bool)
        if not np.any(event_mask):
            return float("nan")
        first_event_index = int(np.argmax(event_mask))
        return float(np.asarray(self.times, dtype=float)[first_event_index])

    def _first_alarm_time(self) -> float:
        event_start_time = self._event_start_time()
        eligible = self.alarms & (np.asarray(self.times) >= event_start_time)
        if np.isnan(event_start_time) or not np.any(eligible):
            return float("nan")
        first_alarm_index = int(np.argmax(eligible))
        return float(np.asarray(self.times, dtype=float)[first_alarm_index])

    def summary(self) -> dict[str, float | int]:
        true_positives = int(np.sum(self.alarms & self.event_flags))
        false_positives = int(np.sum(self.alarms & ~self.event_flags))
        false_negatives = int(np.sum(~self.alarms & self.event_flags))
        true_negatives = int(np.sum(~self.alarms & ~self.event_flags))
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0

        #recall, also called probability of detection
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0

        #FAR, False Alarm Rate
        false_alarm_rate = false_positives / (false_positives + true_negatives) if false_positives + true_negatives else 0.0

        #accuracy
        accuracy = (true_positives + true_negatives) / len(self.alarms) if len(self.alarms) else 0.0

        event_start_time = self._event_start_time()
        first_alarm_time = self._first_alarm_time()
        if np.isnan(event_start_time) or np.isnan(first_alarm_time):
            detection_latency_seconds = float("nan")
        else:
            detection_latency_seconds = first_alarm_time - event_start_time

        return {
            "precision": precision,
            "accuracy": accuracy,
            "probability_of_detection": recall,
            "recall": recall,
            "false_alarm_rate": false_alarm_rate,
            "detection_latency_seconds": detection_latency_seconds,
            "alarms": int(np.sum(self.alarms)),
        }
