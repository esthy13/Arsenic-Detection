"""GRASP with path relinking for ANN architecture and sensor selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Architecture = tuple[int, ...]
MetricEvaluator = Callable[[Architecture], float]
SensorSubset = tuple[int, ...]
SensorSubsetEvaluator = Callable[[SensorSubset], float]


@dataclass(frozen=True)
class GraspConfig:
    iterations: int = 8
    max_evaluations: int = 40
    elite_size: int = 6
    relink_solutions: int = 2
    min_hidden_size: int = 4
    alpha: float = 0.3
    metric_tolerance: float = 0.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.max_evaluations < 1:
            raise ValueError("iterations and max_evaluations must be positive")
        if self.elite_size < 1 or self.relink_solutions < 0:
            raise ValueError("elite_size must be positive and relink_solutions non-negative")
        if self.min_hidden_size < 1:
            raise ValueError("min_hidden_size must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between zero and one")


@dataclass(frozen=True)
class ArchitectureScore:
    hidden_sizes: Architecture
    metric: float
    parameter_count: int
    feasible: bool


@dataclass(frozen=True)
class GraspResult:
    hidden_sizes: Architecture
    metric: float
    baseline_metric: float
    parameter_count: int
    evaluations: int
    elite: tuple[ArchitectureScore, ...]


@dataclass(frozen=True)
class SensorGraspConfig:
    iterations: int = 8
    max_evaluations: int = 40
    elite_size: int = 6
    relink_solutions: int = 2
    alpha: float = 0.3
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.max_evaluations < 1:
            raise ValueError("iterations and max_evaluations must be positive")
        if self.elite_size < 1 or self.relink_solutions < 0:
            raise ValueError("elite_size must be positive and relink_solutions non-negative")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between zero and one")


@dataclass(frozen=True)
class SensorSubsetScore:
    sensor_indices: SensorSubset
    metric: float


@dataclass(frozen=True)
class SensorGraspResult:
    sensor_indices: SensorSubset
    metric: float
    evaluations: int
    elite: tuple[SensorSubsetScore, ...]


def parameter_count(
    input_dim: int, hidden_sizes: Sequence[int], output_dim: int = 1
) -> int:
    """Return the number of trainable weights and biases in a dense ANN."""
    dimensions = (input_dim, *hidden_sizes, output_dim)
    return sum(
        dimensions[index] * dimensions[index + 1] + dimensions[index + 1]
        for index in range(len(dimensions) - 1)
    )


def grasp_architecture_search(
    evaluator: MetricEvaluator,
    *,
    input_dim: int,
    baseline_hidden_sizes: Sequence[int],
    config: GraspConfig = GraspConfig(),
) -> GraspResult:
    """Minimize ANN size while matching the baseline validation metric.

    Construction has no fixed neuron ceiling: an infeasible architecture can
    keep doubling one layer until it becomes feasible. ``max_evaluations`` is a
    runtime safeguard for noisy or unreachable metrics, not a node limit.
    """
    baseline = tuple(int(size) for size in baseline_hidden_sizes)
    if not baseline or any(size < 1 for size in baseline):
        raise ValueError("baseline_hidden_sizes must contain positive integers")

    rng = np.random.default_rng(config.random_seed)
    cache: dict[Architecture, ArchitectureScore] = {}

    baseline_metric = float(evaluator(baseline))
    target_metric = baseline_metric - config.metric_tolerance

    def score(architecture: Architecture) -> ArchitectureScore | None:
        architecture = tuple(max(config.min_hidden_size, int(x)) for x in architecture)
        if architecture in cache:
            return cache[architecture]
        if len(cache) >= config.max_evaluations:
            return None
        metric = float(evaluator(architecture))
        result = ArchitectureScore(
            architecture,
            metric,
            parameter_count(input_dim, architecture),
            metric >= target_metric,
        )
        cache[architecture] = result
        return result

    baseline_score = ArchitectureScore(
        baseline,
        baseline_metric,
        parameter_count(input_dim, baseline),
        True,
    )
    cache[baseline] = baseline_score
    elite: list[ArchitectureScore] = [baseline_score]

    def update_elite(candidate: ArchitectureScore | None) -> None:
        if candidate is None or not candidate.feasible:
            return
        by_architecture = {item.hidden_sizes: item for item in elite}
        by_architecture[candidate.hidden_sizes] = candidate
        elite[:] = sorted(
            by_architecture.values(),
            key=lambda item: (item.parameter_count, -item.metric),
        )[: config.elite_size]

    def construct() -> ArchitectureScore | None:
        current = tuple(config.min_hidden_size for _ in baseline)
        current_score = score(current)
        while current_score is not None and not current_score.feasible:
            candidates: list[ArchitectureScore] = []
            for layer in range(len(current)):
                neighbor = list(current)
                neighbor[layer] *= 2
                candidate = score(tuple(neighbor))
                if candidate is not None:
                    candidates.append(candidate)
            if not candidates:
                return current_score
            utilities = np.asarray(
                [
                    (candidate.metric - current_score.metric)
                    / max(candidate.parameter_count - current_score.parameter_count, 1)
                    for candidate in candidates
                ]
            )
            cutoff = utilities.max() - config.alpha * (
                utilities.max() - utilities.min()
            )
            restricted = [
                candidate
                for candidate, utility in zip(candidates, utilities)
                if utility >= cutoff
            ]
            current_score = restricted[int(rng.integers(len(restricted)))]
            current = current_score.hidden_sizes
        return current_score

    def local_search(candidate: ArchitectureScore) -> ArchitectureScore:
        best = candidate
        improved = True
        while improved:
            improved = False
            neighbors: list[ArchitectureScore] = []
            for layer, size in enumerate(best.hidden_sizes):
                if size <= config.min_hidden_size:
                    continue
                for reduced in {max(config.min_hidden_size, size // 2), size - 1}:
                    architecture = list(best.hidden_sizes)
                    architecture[layer] = reduced
                    result = score(tuple(architecture))
                    if result is not None and result.feasible:
                        neighbors.append(result)
            if neighbors:
                winner = min(
                    neighbors, key=lambda item: (item.parameter_count, -item.metric)
                )
                if winner.parameter_count < best.parameter_count:
                    best, improved = winner, True
        return best

    def path_relink(
        start: ArchitectureScore, guide: ArchitectureScore
    ) -> ArchitectureScore:
        current = list(start.hidden_sizes)
        best = start
        while tuple(current) != guide.hidden_sizes:
            moves: list[ArchitectureScore] = []
            for layer, target in enumerate(guide.hidden_sizes):
                if current[layer] == target:
                    continue
                moved = current.copy()
                delta = target - moved[layer]
                moved[layer] += int(np.sign(delta)) * max(1, abs(delta) // 2)
                if (delta > 0 and moved[layer] > target) or (
                    delta < 0 and moved[layer] < target
                ):
                    moved[layer] = target
                result = score(tuple(moved))
                if result is not None:
                    moves.append(result)
            if not moves:
                break
            next_score = max(
                moves,
                key=lambda item: (
                    item.feasible,
                    item.metric,
                    -item.parameter_count,
                ),
            )
            current = list(next_score.hidden_sizes)
            if next_score.feasible and (
                next_score.parameter_count < best.parameter_count
                or (
                    next_score.parameter_count == best.parameter_count
                    and next_score.metric > best.metric
                )
            ):
                best = next_score
        return local_search(best)

    for iteration in range(config.iterations):
        if len(cache) >= config.max_evaluations:
            break
        candidate = construct()
        if candidate is None or not candidate.feasible:
            continue
        candidate = local_search(candidate)
        if iteration >= 1 and elite:
            guides = rng.choice(
                len(elite), size=min(config.relink_solutions, len(elite)), replace=False
            )
            for guide_index in np.atleast_1d(guides):
                update_elite(path_relink(candidate, elite[int(guide_index)]))
        update_elite(candidate)

    best = min(elite, key=lambda item: (item.parameter_count, -item.metric))
    return GraspResult(
        hidden_sizes=best.hidden_sizes,
        metric=best.metric,
        baseline_metric=baseline_metric,
        parameter_count=best.parameter_count,
        evaluations=len(cache),
        elite=tuple(elite),
    )


def grasp_sensor_subset_search(
    evaluator: SensorSubsetEvaluator,
    *,
    sensor_count: int,
    max_sensors: int,
    config: SensorGraspConfig = SensorGraspConfig(),
) -> SensorGraspResult:
    """Maximize validation performance using at most ``max_sensors`` sensors."""
    if sensor_count < 1:
        raise ValueError("sensor_count must be positive")
    if not 1 <= max_sensors <= sensor_count:
        raise ValueError("max_sensors must be between one and sensor_count")

    rng = np.random.default_rng(config.random_seed)
    cache: dict[SensorSubset, SensorSubsetScore] = {}
    elite: list[SensorSubsetScore] = []

    def score(indices: Sequence[int]) -> SensorSubsetScore | None:
        subset = tuple(sorted(int(index) for index in indices))
        if (
            not subset
            or len(subset) > max_sensors
            or len(set(subset)) != len(subset)
            or subset[0] < 0
            or subset[-1] >= sensor_count
        ):
            raise ValueError("Invalid sensor subset")
        if subset in cache:
            return cache[subset]
        if len(cache) >= config.max_evaluations:
            return None
        result = SensorSubsetScore(subset, float(evaluator(subset)))
        cache[subset] = result
        return result

    def ranking(item: SensorSubsetScore) -> tuple[float, int, SensorSubset]:
        return (-item.metric, len(item.sensor_indices), item.sensor_indices)

    def update_elite(candidate: SensorSubsetScore | None) -> None:
        if candidate is None:
            return
        by_subset = {item.sensor_indices: item for item in elite}
        by_subset[candidate.sensor_indices] = candidate
        elite[:] = sorted(by_subset.values(), key=ranking)[: config.elite_size]

    def choose_rcl(candidates: list[SensorSubsetScore]) -> SensorSubsetScore:
        metrics = np.asarray([candidate.metric for candidate in candidates])
        cutoff = metrics.max() - config.alpha * (metrics.max() - metrics.min())
        restricted = [
            candidate
            for candidate in candidates
            if candidate.metric >= cutoff
        ]
        return restricted[int(rng.integers(len(restricted)))]

    def construct() -> SensorSubsetScore | None:
        singleton_candidates = [
            candidate
            for index in rng.permutation(sensor_count)
            if (candidate := score((int(index),))) is not None
        ]
        if not singleton_candidates:
            return None
        current = choose_rcl(singleton_candidates)
        best = current
        while len(current.sensor_indices) < max_sensors:
            additions = [
                candidate
                for index in rng.permutation(sensor_count)
                if index not in current.sensor_indices
                and (
                    candidate := score((*current.sensor_indices, int(index)))
                )
                is not None
            ]
            if not additions:
                break
            current = choose_rcl(additions)
            if ranking(current) < ranking(best):
                best = current
        return best

    def local_search(candidate: SensorSubsetScore) -> SensorSubsetScore:
        best = candidate
        while True:
            neighbors: list[SensorSubsetScore] = []
            subset = set(best.sensor_indices)
            if len(subset) > 1:
                for removed in subset:
                    result = score(subset - {removed})
                    if result is not None:
                        neighbors.append(result)
            for removed in subset:
                for added in set(range(sensor_count)) - subset:
                    result = score((subset - {removed}) | {added})
                    if result is not None:
                        neighbors.append(result)
            if len(subset) < max_sensors:
                for added in set(range(sensor_count)) - subset:
                    result = score(subset | {added})
                    if result is not None:
                        neighbors.append(result)
            if not neighbors:
                return best
            winner = min([best, *neighbors], key=ranking)
            if winner.sensor_indices == best.sensor_indices:
                return best
            best = winner

    def path_relink(
        start: SensorSubsetScore, guide: SensorSubsetScore
    ) -> SensorSubsetScore:
        current = set(start.sensor_indices)
        target = set(guide.sensor_indices)
        best = start
        while current != target:
            moves: list[SensorSubsetScore] = []
            for removed in current - target:
                moved = current - {removed}
                if moved:
                    result = score(moved)
                    if result is not None:
                        moves.append(result)
            for added in target - current:
                moved = current | {added}
                if len(moved) <= max_sensors:
                    result = score(moved)
                    if result is not None:
                        moves.append(result)
            if not moves:
                break
            next_score = min(moves, key=ranking)
            current = set(next_score.sensor_indices)
            if ranking(next_score) < ranking(best):
                best = next_score
        return local_search(best)

    for iteration in range(config.iterations):
        if len(cache) >= config.max_evaluations:
            break
        candidate = construct()
        if candidate is None:
            break
        candidate = local_search(candidate)
        if iteration and elite:
            guide_indices = rng.choice(
                len(elite),
                size=min(config.relink_solutions, len(elite)),
                replace=False,
            )
            for guide_index in np.atleast_1d(guide_indices):
                update_elite(path_relink(candidate, elite[int(guide_index)]))
        update_elite(candidate)

    if not elite:
        if not cache:
            raise RuntimeError("Sensor GRASP could not evaluate any subset")
        elite = [min(cache.values(), key=ranking)]
    best = min(elite, key=ranking)
    return SensorGraspResult(
        sensor_indices=best.sensor_indices,
        metric=best.metric,
        evaluations=len(cache),
        elite=tuple(elite),
    )
