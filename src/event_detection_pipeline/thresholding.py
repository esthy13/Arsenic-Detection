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
    """Calculate filtered rolling thresholds in linear time.

    The previous implementation sliced every window and recomputed its mean and
    standard deviation from scratch.  Cumulative counts, sums and squared sums
    produce the same population statistics without the nested O(n * window)
    work.  Float64 accumulators reduce cancellation before returning float32,
    as expected by the rest of the pipeline.
    """
    residuals = np.asarray(residuals, dtype=np.float32).reshape(-1)
    if residuals.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    window_size = max(1, int(round(len(residuals) * window_fraction)))
    window_size = min(window_size, len(residuals))

    valid = (residuals <= outlier_upper) & (residuals >= outlier_lower)
    values = np.where(valid, residuals, 0.0).astype(np.float64, copy=False)

    cumulative_count = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    cumulative_sum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    cumulative_squared_sum = np.concatenate(
        ([0.0], np.cumsum(values * values, dtype=np.float64))
    )

    counts = cumulative_count[window_size:] - cumulative_count[:-window_size]
    if np.any(counts == 0):
        # Preserve the previous implementation's behavior for an empty filtered
        # window: all thresholds become zero.
        zeros = np.zeros(len(residuals), dtype=np.float32)
        return zeros, zeros.copy()

    sums = cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    squared_sums = (
        cumulative_squared_sum[window_size:]
        - cumulative_squared_sum[:-window_size]
    )
    means = sums / counts
    variances = np.maximum(squared_sums / counts - means * means, 0.0)
    standard_deviations = np.sqrt(variances)

    rolling_upper = means + upper_multiplier * standard_deviations
    rolling_lower = means - lower_multiplier * standard_deviations
    prefix_size = window_size - 1
    upper = np.concatenate(
        (np.full(prefix_size, rolling_upper[0]), rolling_upper)
    ).astype(np.float32)
    lower = np.concatenate(
        (np.full(prefix_size, rolling_lower[0]), rolling_lower)
    ).astype(np.float32)

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
    false_positive_rate = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0

    # penalty = 1.0 if np.all(outliers == outliers[0]) else 0.0
    # objective = -((true_positive + true_negative) / len(event_flags)) + penalty

    specificity = true_negative / (true_negative + false_positive)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    objective = -balanced_accuracy
    return objective, sensitivity, false_positive_rate, upper, lower, outliers


def _config_from_genes(genes: np.ndarray) -> ThresholdConfig:
    return ThresholdConfig(
        window_size_seconds=0.0,
        window_fraction=float(genes[0]),
        upper_multiplier=float(genes[1]),
        lower_multiplier=float(genes[2]),
        outlier_upper=float(genes[3]),
        outlier_lower=-float(genes[4]),
    )


def _random_genes(rng: np.random.Generator) -> np.ndarray:
    return np.asarray(
        [
            rng.uniform(0.03, 0.2),
            rng.uniform(0.5, 4.0),
            rng.uniform(0.5, 4.0),
            rng.uniform(1.5, 5.0),
            rng.uniform(1.5, 5.0),
        ],
        dtype=np.float32,
    )


def _clip_genes(genes: np.ndarray) -> np.ndarray:
    bounds = np.asarray(
        [
            [0.03, 0.2],
            [0.5, 4.0],
            [0.5, 4.0],
            [1.5, 5.0],
            [1.5, 5.0],
        ],
        dtype=np.float32,
    )
    return np.clip(genes, bounds[:, 0], bounds[:, 1])


def fit_threshold(
    residuals: np.ndarray,
    event_flags: np.ndarray,
    *,
    population_size: int = 24,
    generations: int = 24,
    elite_count: int = 6,
) -> tuple[ThresholdConfig, float, float, np.ndarray, np.ndarray]:
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if not 1 <= elite_count <= population_size:
        raise ValueError("elite_count must be between 1 and population_size")

    rng = np.random.default_rng(42)

    population = [_random_genes(rng) for _ in range(population_size)]
    best_objective = float("inf")
    best_config = _config_from_genes(population[0])
    best_tp = 0.0
    best_fp = 0.0
    best_upper = np.zeros_like(residuals)
    best_lower = np.zeros_like(residuals)

    for _ in range(generations):
        scored: list[tuple[float, np.ndarray, float, float, np.ndarray, np.ndarray]] = []
        for genes in population:
            config = _config_from_genes(genes)
            objective, sensitivity, false_positive_rate, upper, lower, _ = evaluate_threshold(residuals, event_flags, config)
            scored.append((objective, genes, sensitivity, false_positive_rate, upper, lower))
            if objective < best_objective:
                best_objective = objective
                best_config = config
                best_tp = sensitivity
                best_fp = false_positive_rate
                best_upper = upper
                best_lower = lower

        scored.sort(key=lambda item: item[0])
        elites = [genes for _, genes, _, _, _, _ in scored[:elite_count]]
        next_population = elites.copy()

        while len(next_population) < population_size:
            parent_a, parent_b = rng.choice(elites, size=2, replace=True)
            crossover_mask = rng.random(parent_a.shape) < 0.5
            child = np.where(crossover_mask, parent_a, parent_b)
            mutation = rng.normal(0.0, [0.02, 0.35, 0.35, 0.35, 0.35], size=child.shape)
            next_population.append(_clip_genes(child + mutation))

        population = next_population

    return best_config, best_tp, best_fp, best_upper, best_lower
