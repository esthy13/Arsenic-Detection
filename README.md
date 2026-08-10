# Arsenic Detection in Water Distribution Networks

This project was completed as part of the **Machine Learning for Water Distribution Systems** course at the University of Bielefeld (Summer Semester 2025-2026).

## Overview

This project investigates machine-learning-based methods for detecting arsenic contamination in drinking water distribution systems. While traditional chemical analysis is commonly used for arsenic detection, this work explores an alternative approach using data-driven models that leverage water quality measurements from distributed sensor networks.

Arsenic contamination in drinking water poses severe health risks, including nausea, gastrointestinal disorders, skin lesions, cardiovascular diseases, and several types of cancer. Early and reliable detection is essential for protecting public health.

The main contribution of this work is an integrated pipeline that combines **Artificial Neural Network (ANN)-based arsenic contamination detection** with **GRASP-based sensor optimization**. The ANN identifies contamination events from chlorine concentration measurements, while a Greedy Randomized Adaptive Search Procedure (GRASP) optimizes sensor placement to reduce monitoring infrastructure costs without significantly sacrificing detection performance.

## Key Results

- **Best ANN Performance:** Balanced Accuracy of 70.46%, with a specificity of 77.44% and detection latency of 37.5 minutes for arsenic contamination events
- **Chlorine Prediction Accuracy:** R² ≈ 0.91, indicating that chlorine concentrations can be modeled accurately with neural networks
- **Sensor Optimization:** GRASP successfully identified effective sensor configurations with only 3 chlorine sensors (instead of the full network deployment) while maintaining competitive detection performance
- **Baseline Comparison:** The linear regression baseline proved insufficient for reliable contamination detection, demonstrating the superiority of nonlinear machine-learning models

## Repository Structure

```
.
├── README.md                           # This file
├── pyproject.toml                      # Project configuration and dependencies
├── main.py                             # Placeholder entry point
├── src/
│   ├── linear_predictor.py            # Legacy linear regression detector (baseline)
│   ├── linear_predictor_functions.py  # Helper functions for linear model
│   ├── GRASP_linear_predictor.py      # Sensor optimization for linear model
│   ├── data/                           # (Data files, should contain .npz simulation outputs)
│   ├── notebooks/                      # Jupyter notebooks for exploration and visualization
│   │   └── net1_visualization.ipynb    # Network topology and simulation visualization
│   ├── results/                        # Trained models and detection results
│   │   ├── tuned_detector/            # Baseline ANN detector model
│   │   └── supervised_ann_detector/   # Final supervised ANN event detector
│   └── event_detection_pipeline/       # Main detection pipeline module
│       ├── __init__.py                 # Package exports
│       ├── cli.py                      # Command-line interface for training and inference
│       ├── constants.py                # Configuration constants (history, thresholds, etc.)
│       ├── data.py                     # Data loading, sequence creation, sensor grouping
│       ├── model.py                    # Neural network definitions and training logic
│       ├── pipeline.py                 # Main detection pipeline and integration
│       ├── evaluation.py               # Metrics, visualization, and result aggregation
│       ├── thresholding.py             # Threshold calibration and Bayesian probability
│       ├── classes.py                  # Data classes (DatasetSplit, DetectionResult, etc.)
│       ├── grasp.py                    # GRASP sensor selection algorithm
│       └── arsenic_contamination.msx   # EPANET-MSX contamination model definition
```

## How to Run

### Prerequisites

- Python 3.12+
- `uv` package manager ([install from here](https://docs.astral.sh/uv/))
- GPU (optional, but recommended for faster training)

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repo-url>
   cd Arsenic-Detection
   ```
2. **Create and activate the virtual environment:**

   ```bash
   uv sync
   ```

   On Linux/macOS:

   ```bash
   source .venv/bin/activate
   ```

   On Windows:

   ```cmd
   .venv\Scripts\activate
   ```
3. **Install additional dependencies (if needed):**

   ```bash
   uv add <library_name>
   ```

### Prepare Data

The project expects water quality simulation data in `.npz` format (NumPy compressed arrays) from the EPyT-Flow package. Each file should contain:

- `sensor_readings`: Array of sensor measurements (chlorine, flow, arsenic concentrations)
- `sensor_readings_time`: Timestamps of measurements (in seconds)
- `col_desc`: Column descriptions identifying sensor types and nodes

**Note:** Data files are not included in this repository. To generate them, you must:

1. Install EPyT-Flow: `uv add epyt-flow`
2. Run water distribution network simulations with contamination scenarios
3. Export sensor readings to `.npz` files and organize them in `src/data/`

See the existing notebooks (e.g., `src/notebooks/net1_visualization.ipynb`) for examples of how to generate and structure data.

### Run the Detection Pipeline

#### Option 1: CLI with Default Settings

Train and test the detector with default parameters:

```bash
uv run python -c 'from src.event_detection_pipeline.cli import main; main()'
```

The default configuration uses:

- 48 samples of half-hour history (one full daily cycle)
- Chlorine concentration as the primary water quality indicator
- AsIII arrival threshold of 0.01 mg/L for event labeling
- 20% of training scenarios reserved for validation
- Balanced accuracy maximization on validation data

#### Option 2: CLI with Custom Parameters

Customize the training pipeline:

```bash
uv run python -c 'from src.event_detection_pipeline.cli import main; main()' \
  --data-dir src/data \
  --history 96 \
  --epochs 100 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --max-validation-false-alarm-rate 0.2 \
  --dropout 0.1 \
  --weight-decay 1e-4
```

**Common CLI Options:**

- `--data-dir`: Path to directory containing `.npz` data files
- `--history`: Number of historical samples to use (default: 48)
- `--epochs`: Maximum training epochs (default: 80)
- `--batch-size`: Batch size for training (default: 128)
- `--learning-rate`: Optimizer learning rate (default: 1e-3)
- `--arsenic-threshold`: Concentration threshold for labeling events (default: 0.01)
- `--validation-fraction`: Fraction of scenarios reserved for validation (default: 0.2)
- `--max-validation-false-alarm-rate`: Upper limit on false alarms during calibration (default: 0.2)
- `--dropout`: Dropout rate for regularization (default: 0.1)
- `--weight-decay`: L2 regularization strength (default: 1e-4)
- `--early-stopping-patience`: Epochs to wait for improvement before stopping (default: 12)

#### Option 3: Programmatic Usage

Use the pipeline directly in Python:

```python
from pathlib import Path
from src.event_detection_pipeline import run_pipeline, build_detector

# Train detector on all available data
data_dir = Path("src/data")
report = run_pipeline(
    data_dir=data_dir,
    history=48,
    epochs=80,
    batch_size=128,
    learning_rate=1e-3,
)
print(report)

# Or build detector and use it for inference
train_paths = [...]  # List of Path objects to training .npz files
detector = build_detector(train_paths, history=48, epochs=80)
result = detector.detect(Path("path/to/test_file.npz"))
print(f"Balanced Accuracy: {result.summary()['balanced_accuracy']}")
```

### Output

The pipeline produces a JSON report with the following structure:

```json
{
  "device": "cuda" or "cpu",
  "train": {
    "precision": 0.0,
    "recall": 0.0,
    "specificity": 0.0,
    "balanced_accuracy": 0.0,
    "false_alarm_rate": 0.0,
    "detection_latency_seconds": 0.0
  },
  "test": { ... },
  "threshold": { ... },
  "calibration": { ... }
}
```

## Architecture and Pipeline

### Data Flow

1. **Data Loading:** Load `.npz` simulation files containing chlorine, flow, and arsenic sensor measurements from the Net1 water distribution network
2. **Sequence Creation:** Create supervised sequences using a sliding window of historical chlorine/flow data, with each sequence labeled by arsenic arrival
3. **Train/Validation Split:** Randomly partition simulation scenarios into training and validation folds
4. **Model Training:** Train separate ANN models for each chlorine sensor node, using mean-squared error loss and early stopping
5. **Threshold Calibration:** Fit detection thresholds on validation residuals to maximize sensitivity while keeping false-alarm rate below a limit
6. **Inference:** Use trained models to detect contamination in test scenarios and aggregate results across multiple sensor nodes

### Core Components

**`WaterQualityANN` (model.py):**

- Multi-layer feedforward neural network with GELU activations and dropout regularization
- Predicts chlorine concentrations from historical flow and chlorine measurements
- Residuals (actual − predicted chlorine) indicate anomalies caused by arsenic contamination

**`EventClassificationANN` (model.py):**

- Supervised binary classifier trained on contamination events
- Uses multiscale temporal features (current flows, chlorine changes over 1/3/6/24/48 samples) and cyclical time-of-day encoding
- Outputs event probability used for final alarm decisions

**`EventDetector` (pipeline.py):**

- Manages multiple trained ANN models (one per sensor node)
- Combines residual-based and supervised classification approaches
- Generates detection results with alarms, probabilities, and diagnostics

**GRASP Sensor Optimization (grasp.py):**

- Greedy randomized algorithm for selecting a subset of chlorine sensors
- Balances detection performance against infrastructure cost
- Identified 3-sensor configurations maintaining competitive accuracy

## Notes on Model Defaults

The ANN detector is configured with the following defaults:

- **History:** 48 half-hour samples (one complete 24-hour demand cycle)
- **Input Features:**
  - Flow measurements from all nodes
  - Historical chlorine concentrations
  - Cyclical time-of-day encoding (sine/cosine of time-of-day)
  - *Deliberately excludes* contemporaneous chlorine from other nodes to prevent feature leakage
- **Event Labels:** Aligned per chlorine target node using AsIII arrival threshold of 0.01 mg/L
- **Alarm Calibration:** Maximizes validation sensitivity while keeping false-alarm rate below 10% (default)

To retrain a detector with modified feature layouts, saved models must be discarded and the pipeline re-run.

## Limitations and Future Work

- **Limited Single Indicator:** The framework relies only on chlorine concentration as the water quality indicator, whereas other approaches use multiple indicators. Performance could improve with additional measurements
- **Recall Gap:** Current detection recall (44.97%–63.47%) reflects the restrictive sensing conditions (single indicator), but leaves room for improvement through additional data integration
- **Future Directions:**
  - Evaluate on larger, more realistic water distribution networks
  - Investigate additional contamination scenarios and types
  - Explore advanced architectures (e.g., LSTMs, graph neural networks for network topology)
  - Incorporate additional hydraulic and water-quality indicators
  - Reduce false-alarm rates through improved threshold calibration

## License

This project is licensed under the GNU General Public License v3.0 (see LICENSE file).
