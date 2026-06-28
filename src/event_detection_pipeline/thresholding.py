import numpy as np
from .classes import ThresholdConfig
# dynamic thresholding based on the paper pseudocode
# Pseudocode source: https://www.sciencedirect.com/science/article/pii/S0043135413000341#:~:text=Download%20all%20supplementary%20files%20included%20with%20this%20article

def rolling_thresholds(
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


def bayesian_event_probability(
    residuals: np.ndarray,
    upper_threshold: np.ndarray,
    lower_threshold: np.ndarray,
    true_positive_rate: float,
    false_positive_rate: float,
    *,
    initial_probability: float = 1e-5,
    smoothing: float = 0.6,
) -> np.ndarray:
    probability = initial_probability
    event_probability = np.zeros(len(residuals), dtype=np.float32)

    for index, residual in enumerate(residuals):
        if residual > upper_threshold[index] or residual < lower_threshold[index]:
            previous_probability = probability
            denominator = true_positive_rate * probability + false_positive_rate * (1 - probability)
            if denominator > 0:
                probability = true_positive_rate * probability / denominator
            probability = smoothing * probability + (1 - smoothing) * previous_probability
            probability = min(probability, 0.95)
        else:
            previous_probability = probability
            denominator = (1 - true_positive_rate) * probability + (1 - false_positive_rate) * (1 - probability)
            if denominator > 0:
                probability = (1 - true_positive_rate) * probability / denominator
            probability = smoothing * probability + (1 - smoothing) * previous_probability
            probability = max(probability, initial_probability)
        event_probability[index] = probability

    return event_probability


def evaluate_threshold(
    residuals: np.ndarray,
    event_flags: np.ndarray,
    config: ThresholdConfig,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    upper, lower = rolling_thresholds(
        residuals,
        config.window_fraction,
        config.upper_multiplier,
        config.lower_multiplier,
        config.outlier_upper,
        config.outlier_lower,
    )
    outliers = (residuals > upper) | (residuals < lower)

    true_positive = int(np.sum(outliers & event_flags))
    false_positive = int(np.sum(outliers & ~event_flags))
    false_negative = int(np.sum(~outliers & event_flags))
    true_negative = int(np.sum(~outliers & ~event_flags))

    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    false_positive_rate = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0

    penalty = 1.0 if np.all(outliers == outliers[0]) else 0.0
    objective = -(sensitivity + specificity) + penalty
    return objective, sensitivity, false_positive_rate, upper, lower, outliers


def fit_threshold(
    residuals: np.ndarray,
    event_flags: np.ndarray,
) -> tuple[ThresholdConfig, float, float, np.ndarray, np.ndarray]:
    candidate_windows = (0.03, 0.05, 0.08, 0.1, 0.15)
    candidate_multipliers = (0.5, 1.0, 1.5, 2.0)
    candidate_filters = (1.5, 2.0, 3.0)

    best_objective = float("inf")
    best_config = ThresholdConfig(0.05, 1.0, 1.0, 3.0, -3.0)
    best_tp = 0.0
    best_fp = 0.0
    best_upper = np.zeros_like(residuals)
    best_lower = np.zeros_like(residuals)

    for window_fraction in candidate_windows:
        for upper_multiplier in candidate_multipliers:
            for lower_multiplier in candidate_multipliers:
                for filter_scale in candidate_filters:
                    config = ThresholdConfig(
                        window_fraction=window_fraction,
                        upper_multiplier=upper_multiplier,
                        lower_multiplier=lower_multiplier,
                        outlier_upper=filter_scale,
                        outlier_lower=-filter_scale,
                    )
                    objective, sensitivity, false_positive_rate, upper, lower, _ = evaluate_threshold(residuals, event_flags, config)
                    if objective < best_objective:
                        best_objective = objective
                        best_config = config
                        best_tp = sensitivity
                        best_fp = false_positive_rate
                        best_upper = upper
                        best_lower = lower

    return best_config, best_tp, best_fp, best_upper, best_lower
