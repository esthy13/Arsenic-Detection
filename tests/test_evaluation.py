import unittest

import numpy as np

from src.event_detection_pipeline.classes import DetectionResult
from src.event_detection_pipeline.evaluation import aggregate_evaluation_summary


class AggregateEvaluationSummaryTests(unittest.TestCase):
    def test_reports_classifier_and_regression_metrics(self) -> None:
        result = DetectionResult(
            times=np.arange(4, dtype=float),
            residuals=np.zeros((4, 1)),
            upper_threshold=np.ones((4, 1)),
            lower_threshold=-np.ones((4, 1)),
            event_probability=np.array([0.1, 0.8, 0.4, 0.9]),
            alarms=np.array([False, True, False, True]),
            event_flags=np.array([False, True, True, True]),
            predicted=np.array([[0.0], [1.0], [2.0], [4.0]]),
            actual=np.array([[0.0], [1.0], [3.0], [3.0]]),
            parameter_event_probability=np.zeros((4, 1)),
            parameter_alarms=np.zeros((4, 1), dtype=bool),
        )

        summary = aggregate_evaluation_summary([result])

        self.assertAlmostEqual(summary["precision"], 1.0)
        self.assertAlmostEqual(summary["recall"], 2 / 3)
        self.assertAlmostEqual(summary["specificity"], 1.0)
        self.assertAlmostEqual(summary["balanced_accuracy"], 5 / 6)
        self.assertAlmostEqual(summary["false_alarm_rate"], 0.0)
        self.assertAlmostEqual(summary["chlorine_mae"], 0.5)
        self.assertAlmostEqual(summary["chlorine_rmse"], np.sqrt(0.5))
        self.assertIn("roc_auc", summary)
        self.assertIn("average_precision", summary)


if __name__ == "__main__":
    unittest.main()
