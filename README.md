# Arsenic-Detection
Data-driven (arsenic) contamination detection project for the course Machine Learning for Water Distribution System at the university of Bielefeld (a.a. 2025-2026)

# How to run the project
First verify you have a version of python 3.12 and uv package manager installed on your device.
0. On your first time create a virtual environment
    ```bash
    uv sync
    ```
1. Activate the virtual environment .venv
    on linux:
    ```bash
    source .venv/bin/activate
    ```
    on windows:
    ```ps
    .venv\Scripts\activate
    ```

## Remember!
The correct way to install a new libraries on your local environment is:
```bash
uv add <library_name>
```

# Link to the report

https://unibielefeldde-my.sharepoint.com/:w:/r/personal/julian_wick_uni-bielefeld_de/Documents/Dokumente/Uni/Master/Semester%208/WDN/Report.docx?d=wc81b8dfa5fde43c49d45f7bd3dd66571&csf=1&web=1&e=qlJz6v

## Event detector defaults

The ANN detector uses one day of half-hour history (`history=48`), current flow
measurements, and sine/cosine time-of-day features. Current chlorine values from
other nodes are deliberately excluded to prevent contemporaneous feature
leakage.

By default, event labels are aligned per chlorine target node using an AsIII
arrival threshold of 0.01. Training scenarios are split again into model-fit and
validation scenarios; dynamic thresholds and the final alarm cutoff are
calibrated on validation only. The cutoff maximizes validation sensitivity while
keeping its false-alarm rate below 10%.

Useful alternatives can be tested from the CLI:

```bash
uv run python -c 'from src.event_detection_pipeline.cli import main; main()' \
  --data-dir src/data --history 96 --max-validation-false-alarm-rate 0.2
```

Saved detectors produced by the previous feature layout must be retrained before
they can be loaded by the updated pipeline.

### Tuned ANN configuration

A validation-only search selected the following configuration:

```text
hidden sizes: (64, 32)
learning rate: 3e-4
dropout: 0.0
weight decay: 1e-4
batch size: 128
maximum epochs: 60, with early stopping patience 8
validation false-alarm limit: 0.2
```

The saved detector is in `src/results/tuned_detector`. On the held-out scenarios
17-20 it achieved balanced accuracy 0.564, recall 0.310, false-alarm rate 0.181,
and precision 0.115.

### Supervised ANN event head

The final detector keeps the chlorine-prediction ANN and residual thresholds for
diagnostics, but uses a supervised binary ANN for the actual event probability.
Its inputs are current flows and chlorine, multiscale chlorine changes over
1, 3, 6, 24, and 48 samples, and cyclical time-of-day features. Class-weighted
binary cross entropy handles the event imbalance.

The event head uses hidden sizes `(128, 64)`, learning rate `3e-4`, dropout `0.1`,
and early stopping. The saved model is in `src/results/supervised_ann_detector`.
On held-out scenarios 17-20 it achieved balanced accuracy 0.687, recall 0.561,
false-alarm rate 0.187, and precision 0.185.
