"""Train and compare matched ANN regressor/classifier architectures."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.event_detection import (
    aggregate_evaluation_summary,
    build_detector,
    collect_default_data_paths,
    collect_strong_contamination_paths,
    split_train_test,
)
from src.event_detection_pipeline.model_io import save_detector


ARCHITECTURES = ((32,), (64, 32), (128, 64), (128, 64, 32))


def evaluate_dataset(
    name: str,
    paths: list[Path],
    output_dir: Path,
) -> list[dict[str, float | str]]:
    train_paths, test_paths = split_train_test(paths)
    summaries: list[dict[str, float | str]] = []

    for architecture in ARCHITECTURES:
        label = "-".join(map(str, architecture))
        print(f"\n[{name}] training matched architecture {label}", flush=True)
        detector = build_detector(
            train_paths,
            history=48,
            epochs=60,
            batch_size=128,
            learning_rate=3e-4,
            hidden_sizes=architecture,
            dropout=0.0,
            weight_decay=1e-4,
            early_stopping_patience=8,
            max_validation_false_alarm_rate=0.2,
            classifier_hidden_sizes=architecture,
        )
        save_detector(
            detector,
            directory=str(output_dir / f"detector_joint_{name}_{label}"),
        )
        results = [detector.detect(path) for path in test_paths]
        summary = aggregate_evaluation_summary(results)
        summary["dataset"] = name
        summary["architecture"] = label
        summaries.append(summary)
        print(pd.Series(summary).to_string(), flush=True)

    return summaries


def main() -> None:
    root = ROOT
    data_dir = root / "src" / "data"
    output_dir = root / "results" / "ANN"
    summaries = [
        *evaluate_dataset(
            "weak",
            collect_default_data_paths(data_dir),
            output_dir,
        ),
        *evaluate_dataset(
            "strong",
            collect_strong_contamination_paths(data_dir),
            output_dir,
        ),
    ]
    comparison = pd.DataFrame(summaries).set_index(["dataset", "architecture"])
    csv_path = output_dir / "joint_architecture_comparison.csv"
    comparison.to_csv(csv_path)
    print(f"\nComplete comparison saved to {csv_path}", flush=True)
    print(comparison.to_string(), flush=True)


if __name__ == "__main__":
    main()
