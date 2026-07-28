import unittest

from src.event_detection_pipeline.grasp import (
    GraspConfig,
    SensorGraspConfig,
    grasp_architecture_search,
    grasp_sensor_subset_search,
    parameter_count,
)


class GraspArchitectureSearchTests(unittest.TestCase):
    def test_parameter_count_includes_weights_and_biases(self) -> None:
        self.assertEqual(parameter_count(3, (4, 2)), 29)

    def test_returns_smaller_architecture_that_matches_baseline(self) -> None:
        def evaluator(hidden_sizes: tuple[int, ...]) -> float:
            # The target is reached as soon as the network has 16 hidden units.
            return min(sum(hidden_sizes) / 16.0, 1.0)

        result = grasp_architecture_search(
            evaluator,
            input_dim=5,
            baseline_hidden_sizes=(16, 8),
            config=GraspConfig(
                iterations=5,
                max_evaluations=60,
                min_hidden_size=2,
                random_seed=7,
            ),
        )

        self.assertGreaterEqual(result.metric, result.baseline_metric)
        self.assertLess(
            result.parameter_count, parameter_count(5, (16, 8))
        )
        self.assertLessEqual(result.evaluations, 60)

    def test_falls_back_to_baseline_when_budget_cannot_find_feasible_model(self) -> None:
        baseline = (8, 4)

        def evaluator(hidden_sizes: tuple[int, ...]) -> float:
            return 1.0 if hidden_sizes == baseline else 0.0

        result = grasp_architecture_search(
            evaluator,
            input_dim=3,
            baseline_hidden_sizes=baseline,
            config=GraspConfig(iterations=2, max_evaluations=3),
        )
        self.assertEqual(result.hidden_sizes, baseline)

    def test_sensor_grasp_finds_bounded_high_scoring_subset(self) -> None:
        calls: list[tuple[int, ...]] = []

        def evaluator(indices: tuple[int, ...]) -> float:
            calls.append(indices)
            score = sum({1: 0.5, 3: 0.3, 4: 0.2}.get(index, 0.0) for index in indices)
            return score

        result = grasp_sensor_subset_search(
            evaluator,
            sensor_count=6,
            max_sensors=2,
            config=SensorGraspConfig(
                iterations=6,
                max_evaluations=30,
                random_seed=4,
            ),
        )

        self.assertEqual(result.sensor_indices, (1, 3))
        self.assertLessEqual(len(result.sensor_indices), 2)
        self.assertLessEqual(result.evaluations, 30)
        self.assertEqual(len(calls), len(set(calls)))

    def test_sensor_grasp_rejects_budget_above_sensor_count(self) -> None:
        with self.assertRaises(ValueError):
            grasp_sensor_subset_search(
                lambda indices: 0.0,
                sensor_count=3,
                max_sensors=4,
            )


if __name__ == "__main__":
    unittest.main()
