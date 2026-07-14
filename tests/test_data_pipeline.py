import unittest

import numpy as np

from src.event_detection_pipeline.classes import SensorGroups
from src.event_detection_pipeline.data import (
    arsenic_arrival_flags,
    event_flags_from_time,
    make_supervised_sequences,
)
from src.event_detection_pipeline.pipeline import (
    _calibrate_alarm_threshold,
    _calibrate_balanced_accuracy_threshold,
    _classification_features,
    _fuse_probabilities,
)


class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = SensorGroups(
            chlorine=[2, 3],
            flow=[0, 1],
            arsenic=[4, 5],
            chlorine_nodes=["10", "11"],
            arsenic_nodes=["10", "11"],
        )

    def test_arsenic_labels_are_aligned_per_target_node(self) -> None:
        readings = np.zeros((4, 6), dtype=np.float32)
        readings[1, 4] = 0.02
        readings[2, 5] = 0.03
        flags = arsenic_arrival_flags(readings, self.groups, threshold=0.01)
        np.testing.assert_array_equal(
            flags,
            [[False, False], [True, False], [False, True], [False, False]],
        )

    def test_sequences_use_flows_target_history_and_daily_time(self) -> None:
        readings = np.arange(48, dtype=np.float32).reshape(8, 6)
        times = np.arange(8, dtype=np.float32) * 1800
        flags = np.zeros((8, 2), dtype=bool)
        split = make_supervised_sequences(
            readings, self.groups, history=3, times=times, event_flags=flags
        )
        self.assertEqual(split.inputs.shape, (5, 2, 7))
        np.testing.assert_array_equal(split.inputs[0, 0, :2], readings[3, :2])
        np.testing.assert_array_equal(split.inputs[0, 0, 2:5], readings[:3, 2])
        self.assertEqual(split.target_flags.shape, (5, 2))

    def test_injection_window_is_half_open(self) -> None:
        times = np.asarray([0.0, 10.0, 20.0])
        np.testing.assert_array_equal(
            event_flags_from_time(times, 0.0, 20.0), [True, True, False]
        )

    def test_probability_fusion(self) -> None:
        values = np.asarray([[0.1, 0.5], [0.2, 0.4]], dtype=np.float32)
        np.testing.assert_allclose(_fuse_probabilities(values, "mean"), [0.3, 0.3])
        np.testing.assert_allclose(_fuse_probabilities(values, "max"), [0.5, 0.4])

    def test_alarm_threshold_is_calibrated_on_balanced_accuracy(self) -> None:
        flags = np.asarray([False, False, True, True])
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
        threshold = _calibrate_alarm_threshold(flags, probabilities)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.8)

    def test_alarm_calibration_respects_false_alarm_limit(self) -> None:
        flags = np.asarray([False] * 10 + [True] * 2)
        probabilities = np.asarray(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.65, 0.95]
        )
        threshold = _calibrate_alarm_threshold(
            flags, probabilities, max_false_alarm_rate=0.1
        )
        false_alarm_rate = np.mean(probabilities[:10] >= threshold)
        self.assertLessEqual(false_alarm_rate, 0.1)

    def test_classifier_features_and_balanced_cutoff(self) -> None:
        readings = np.arange(48, dtype=np.float32).reshape(8, 6)
        times = np.arange(8, dtype=np.float32) * 1800
        split = make_supervised_sequences(
            readings, self.groups, 3, times, np.zeros((8, 2), dtype=bool)
        )
        features = _classification_features(split, self.groups, history=3)
        self.assertEqual(features.shape, (5, 16))
        flags = np.asarray([False, False, True, True])
        probability = np.asarray([0.1, 0.2, 0.8, 0.9])
        cutoff = _calibrate_balanced_accuracy_threshold(flags, probability)
        self.assertGreater(cutoff, 0.2)
        self.assertLessEqual(cutoff, 0.8)


if __name__ == "__main__":
    unittest.main()
