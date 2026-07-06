from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import DEFAULT_EVENT_END_SECONDS, DEFAULT_EVENT_START_SECONDS, DEFAULT_HISTORY
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

    for index, row in enumerate(col_desc):
        text = " ".join(str(value) for value in np.ravel(row)).lower()
        if "chlorine" in text:
            chlorine.append(index)
        elif "asiii" in text or "arsenic" in text:
            arsenic.append(index)
        elif "flow" in text:
            flow.append(index)

    if not chlorine or not flow or not arsenic:
        raise ValueError("Could not infer chlorine, flow, and arsenic sensor groups from the dataset metadata")

    return SensorGroups(chlorine=chlorine, flow=flow, arsenic=arsenic)


def event_flags_from_time(times: np.ndarray, start_seconds: float, end_seconds: float) -> np.ndarray:
    return (times >= start_seconds) & (times <= end_seconds)


def make_supervised_sequences(
    readings: np.ndarray,
    groups: SensorGroups,
    history: int,
    times: np.ndarray,
    event_flags: np.ndarray,
) -> DatasetSplit:
    feature_columns = groups.chlorine + groups.flow + groups.arsenic
    target_columns = groups.arsenic

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    flags: list[bool] = []
    aligned_times: list[float] = []

    for time_index in range(history, len(readings)):
        window = readings[time_index - history : time_index, feature_columns]
        inputs.append(window.reshape(-1))
        targets.append(readings[time_index, target_columns])
        flags.append(bool(event_flags[time_index]))
        aligned_times.append(float(times[time_index]))

    return DatasetSplit(
        inputs=np.asarray(inputs, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        flags=np.asarray(flags, dtype=bool),
        times=np.asarray(aligned_times, dtype=np.float32),
    )


def split_file(
    path: Path,
    *,
    history: int = DEFAULT_HISTORY,
    event_start_seconds: float = DEFAULT_EVENT_START_SECONDS,
    event_end_seconds: float = DEFAULT_EVENT_END_SECONDS,
) -> tuple[DatasetSplit, SensorGroups]:
    loaded = load_npz_file(path)
    groups = infer_sensor_groups(loaded["col_desc"])
    flags = event_flags_from_time(loaded["sensor_readings_time"], event_start_seconds, event_end_seconds)
    split = make_supervised_sequences(loaded["sensor_readings"], groups, history, loaded["sensor_readings_time"], flags)
    return split, groups


def collect_default_data_paths(data_dir: Path) -> list[Path]:
    paths = sorted(path for path in data_dir.glob("scada_data_*.npz") if "_no_cont" not in path.name)
    if not paths:
        raise FileNotFoundError(f"No SCADA files found in {data_dir}")
    return paths


def split_train_test(paths: Sequence[Path], train_fraction: float = 0.7) -> tuple[list[Path], list[Path]]:
    split_index = max(1, int(round(len(paths) * train_fraction)))
    if split_index >= len(paths):
        split_index = len(paths) - 1
    return list(paths[:split_index]), list(paths[split_index:])
