from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .constants import (
    DEFAULT_ARSENIC_THRESHOLD,
    DEFAULT_EVENT_END_SECONDS,
    DEFAULT_EVENT_START_SECONDS,
    DEFAULT_HISTORY,
)
from .pipeline import run_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train and run the arsenic contamination event detector")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--event-start-seconds", type=float, default=DEFAULT_EVENT_START_SECONDS)
    parser.add_argument("--event-end-seconds", type=float, default=DEFAULT_EVENT_END_SECONDS)
    parser.add_argument(
        "--label-mode",
        choices=("arsenic_arrival", "injection_window"),
        default="arsenic_arrival",
    )
    parser.add_argument(
        "--arsenic-threshold", type=float, default=DEFAULT_ARSENIC_THRESHOLD
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--fusion-method", choices=("mean", "max", "noisy_or"), default="mean"
    )
    parser.add_argument(
        "--max-validation-false-alarm-rate", type=float, default=0.2
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    args = parser.parse_args(argv)

    report = run_pipeline(
        data_dir=args.data_dir,
        history=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        event_start_seconds=args.event_start_seconds,
        event_end_seconds=args.event_end_seconds,
        label_mode=args.label_mode,
        arsenic_threshold=args.arsenic_threshold,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        fusion_method=args.fusion_method,
        max_validation_false_alarm_rate=args.max_validation_false_alarm_rate,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
    )
    print(json.dumps(report, indent=2))
    return 0
