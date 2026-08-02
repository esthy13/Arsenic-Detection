import unittest

import numpy as np

from src.event_detection_pipeline.thresholding import rolling_thresholds


def _rolling_thresholds_reference(
    residuals: np.ndarray,
    window_fraction: float,
    upper_multiplier: float,
    lower_multiplier: float,
    outlier_upper: float,
    outlier_lower: float,
) -> tuple[np.ndarray, np.ndarray]:
    window_size = max(1, int(round(len(residuals) * window_fraction)))
    upper = np.zeros(len(residuals), dtype=np.float32)
    lower = np.zeros(len(residuals), dtype=np.float32)

    for index in range(window_size, len(residuals) + 1):
        window = residuals[index - window_size : index]
        filtered = window[(window <= outlier_upper) & (window >= outlier_lower)]
        if len(filtered) == 0:
            upper[:] = 0.0
            lower[:] = 0.0
            break
        mean_value = float(np.mean(filtered))
        std_value = float(np.std(filtered))
        upper[index - 1] = mean_value + upper_multiplier * std_value
        lower[index - 1] = mean_value - lower_multiplier * std_value

    if window_size < len(residuals):
        upper[: window_size - 1] = upper[window_size - 1]
        lower[: window_size - 1] = lower[window_size - 1]
    return upper, lower


class RollingThresholdTests(unittest.TestCase):
    def test_fast_rolling_thresholds_match_reference(self) -> None:
        rng = np.random.default_rng(7)
        residuals = rng.normal(0.0, 0.7, 400).astype(np.float32)
        residuals[::31] = 4.0

        arguments = (0.13, 2.1, 1.7, 1.5, -1.5)
        expected = _rolling_thresholds_reference(residuals, *arguments)
        actual = rolling_thresholds(residuals, *arguments)

        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=2e-6)

    def test_empty_filtered_window_preserves_zero_threshold_behavior(self) -> None:
        residuals = np.full(20, 10.0, dtype=np.float32)
        upper, lower = rolling_thresholds(
            residuals, 0.2, 2.0, 2.0, 1.0, -1.0
        )
        np.testing.assert_array_equal(upper, np.zeros_like(residuals))
        np.testing.assert_array_equal(lower, np.zeros_like(residuals))

    def test_single_sample_window(self) -> None:
        residuals = np.asarray([-0.5, 0.25, 0.75], dtype=np.float32)
        upper, lower = rolling_thresholds(
            residuals, 0.01, 2.0, 2.0, 1.0, -1.0
        )
        np.testing.assert_allclose(upper, residuals)
        np.testing.assert_allclose(lower, residuals)


if __name__ == "__main__":
    unittest.main()
