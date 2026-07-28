from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import (
    DEFAULT_ARSENIC_THRESHOLD,
    DEFAULT_EVENT_END_SECONDS,
    DEFAULT_EVENT_START_SECONDS,
    DEFAULT_HISTORY,
    SECONDS_PER_DAY,
)
from .classes import DatasetSplit, SensorGroups


def load_npz_file(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    return {
        "sensor_readings": np.asarray(loaded["sensor_readings"], dtype=np.float32),
        "sensor_readings_time": np.asarray(loaded["sensor_readings_time"], dtype=np.float32).reshape(-1),
        "col_desc": np.asarray(loaded["col_desc"], dtype=object),
    }


def infer_sensor_groups(col_desc: np.ndarray) -> SensorGroups:
    chlorine: list[int] = []
    flow: list[int] = []
    arsenic: list[int] = []
    chlorine_nodes: list[str] = []
    arsenic_nodes: list[str] = []

    for index, row in enumerate(col_desc):
        text = " ".join(str(value) for value in np.ravel(row)).lower()
        if "chlorine" in text:
            chlorine.append(index)
            chlorine_nodes.append(str(np.ravel(row)[1]).split("@")[-1].strip())
        elif "asiii" in text or "arsenic" in text:
            arsenic.append(index)
            arsenic_nodes.append(str(np.ravel(row)[1]).split("@")[-1].strip())
        elif "flow" in text:
            flow.append(index)

    if not chlorine or not flow or not arsenic:
        raise ValueError("Could not infer chlorine, flow, and arsenic sensor groups from the dataset metadata")

    return SensorGroups(
        chlorine=chlorine,
        flow=flow,
        arsenic=arsenic,
        chlorine_nodes=chlorine_nodes,
        arsenic_nodes=arsenic_nodes,
    )


def event_flags_from_time(times: np.ndarray, start_seconds: float, end_seconds: float) -> np.ndarray:
    return (times >= start_seconds) & (times < end_seconds)


def arsenic_arrival_flags(
    readings: np.ndarray,
    groups: SensorGroups,
    threshold: float = DEFAULT_ARSENIC_THRESHOLD,
) -> np.ndarray:
    """Return one physically aligned event flag per chlorine target node."""
    arsenic_by_node = dict(zip(groups.arsenic_nodes, groups.arsenic))
    flags = []
    for node in groups.chlorine_nodes:
        arsenic_column = arsenic_by_node.get(node)
        if arsenic_column is None:
            flags.append(np.zeros(len(readings), dtype=bool))
        else:
            flags.append(readings[:, arsenic_column] >= threshold)
    return np.column_stack(flags)


def make_supervised_sequences(
    readings: np.ndarray,
    groups: SensorGroups,
    history: int,
    times: np.ndarray,
    event_flags: np.ndarray,
) -> DatasetSplit:
    target_columns = groups.chlorine

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    flags: list[bool] = []
    target_flags: list[np.ndarray] = []
    aligned_times: list[float] = []

    for time_index in range(history, len(readings)):
        target_inputs: list[np.ndarray] = []
        time_of_day = float(times[time_index] % SECONDS_PER_DAY) / SECONDS_PER_DAY
        cyclical_time = np.asarray(
            [np.sin(2 * np.pi * time_of_day), np.cos(2 * np.pi * time_of_day)],
            dtype=np.float32,
        )
        for target_column in target_columns:
            current_predictors = readings[time_index, groups.flow]
            lagged_target = readings[time_index - history : time_index, target_column]
            target_inputs.append(
                np.concatenate([current_predictors, lagged_target, cyclical_time])
            )

        inputs.append(np.asarray(target_inputs, dtype=np.float32))
        targets.append(readings[time_index, target_columns])
        current_flags = np.asarray(event_flags[time_index], dtype=bool)
        if current_flags.ndim == 0:
            current_flags = np.repeat(current_flags, len(target_columns))
        target_flags.append(current_flags)
        flags.append(bool(np.any(current_flags)))
        aligned_times.append(float(times[time_index]))

    return DatasetSplit(
        inputs=np.asarray(inputs, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        flags=np.asarray(flags, dtype=bool),
        times=np.asarray(aligned_times, dtype=np.float32),
        target_flags=np.asarray(target_flags, dtype=bool),
    )


def split_file(
    path: Path,
    *,
    history: int = DEFAULT_HISTORY,
    event_start_seconds: float = DEFAULT_EVENT_START_SECONDS,
    event_end_seconds: float = DEFAULT_EVENT_END_SECONDS,
    label_mode: str = "arsenic_arrival",
    arsenic_threshold: float = DEFAULT_ARSENIC_THRESHOLD,
) -> tuple[DatasetSplit, SensorGroups]:
    loaded = load_npz_file(path)
    groups = infer_sensor_groups(loaded["col_desc"])
    if label_mode == "arsenic_arrival":
        flags = arsenic_arrival_flags(
            loaded["sensor_readings"], groups, threshold=arsenic_threshold
        )
    elif label_mode == "injection_window":
        flags = event_flags_from_time(
            loaded["sensor_readings_time"], event_start_seconds, event_end_seconds
        )
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    split = make_supervised_sequences(loaded["sensor_readings"], groups, history, loaded["sensor_readings_time"], flags)
    return split, groups


def collect_default_data_paths(data_dir: Path) -> list[Path]:
    def numeric_suffix(path: Path) -> int:
        suffix = path.stem.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    paths = sorted(
        (
            path
            for path in data_dir.glob("scada_data_*.npz")
            if "_no_cont" not in path.name
            and "_strong_cont" not in path.name
        ),
        key=numeric_suffix,
    )
    if not paths:
        raise FileNotFoundError(f"No SCADA files found in {data_dir}")
    return paths

def collect_strong_contamination_paths(data_dir: Path) -> list[Path]:
    def numeric_suffix(path: Path) -> int:
        suffix = path.stem.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    paths = sorted(
        (
            path
            for path in data_dir.glob("scada_data_*.npz")
            if "_strong_cont" in path.name
        ),
        key=numeric_suffix,
    )
    if not paths:
        raise FileNotFoundError(f"No SCADA files found in {data_dir}")
    return paths


def split_train_test(paths: Sequence[Path], train_fraction: float = 0.8) -> tuple[list[Path], list[Path]]:
    split_index = max(1, int(round(len(paths) * train_fraction)))
    if split_index >= len(paths):
        split_index = len(paths) - 1
    return list(paths[:split_index]), list(paths[split_index:])
