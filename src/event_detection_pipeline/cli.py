from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .constants import DEFAULT_EVENT_END_SECONDS, DEFAULT_EVENT_START_SECONDS, DEFAULT_HISTORY
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
    args = parser.parse_args(argv)

    report = run_pipeline(
        data_dir=args.data_dir,
        history=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        event_start_seconds=args.event_start_seconds,
        event_end_seconds=args.event_end_seconds,
    )
    print(json.dumps(report, indent=2))
    return 0
