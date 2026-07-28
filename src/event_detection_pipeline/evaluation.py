"""
Evaluation metrics for event detection and chlorine prediction.
Includes ROC curves, sensitivity, specificity, and other performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from .classes import DetectionResult


def get_chlorine_regression_summary(
    result: DetectionResult,
    chlorine_indices: list[int] | None = None,
) -> dict[str, float]:
    """Return regression metrics over the selected chlorine channels."""
    if chlorine_indices is None:
        chlorine_indices = list(range(result.actual.shape[1]))
    actual = result.actual[:, chlorine_indices].reshape(-1)
    predicted = result.predicted[:, chlorine_indices].reshape(-1)
    return {
        "chlorine_mae": float(mean_absolute_error(actual, predicted)),
        "chlorine_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "chlorine_r2": float(r2_score(actual, predicted)),
    }


def aggregate_evaluation_summary(
    results: list[DetectionResult],
    chlorine_indices: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate classification and regression after concatenating test scenarios."""
    if not results:
        raise ValueError("At least one DetectionResult is required")

    event_flags = np.concatenate([result.event_flags for result in results])
    alarms = np.concatenate([result.alarms for result in results])
    probabilities = np.concatenate(
        [result.event_probability for result in results]
    )
    actual = np.concatenate([result.actual for result in results], axis=0)
    predicted = np.concatenate([result.predicted for result in results], axis=0)

    y_true = event_flags.astype(bool)
    y_pred = alarms.astype(bool)
    tp = int(np.sum(y_pred & y_true))
    tn = int(np.sum(~y_pred & ~y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if chlorine_indices is None:
        chlorine_indices = list(range(actual.shape[1]))
    chlorine_actual = actual[:, chlorine_indices].reshape(-1)
    chlorine_predicted = predicted[:, chlorine_indices].reshape(-1)

    latencies = np.asarray(
        [result.summary()["detection_latency_seconds"] for result in results],
        dtype=float,
    )
    finite_latencies = latencies[np.isfinite(latencies)]
    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "accuracy": float((tp + tn) / len(y_true)),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "false_alarm_rate": float(1.0 - specificity),
        "detection_latency_seconds": (
            float(np.mean(finite_latencies)) if finite_latencies.size else float("nan")
        ),
        "chlorine_mae": float(
            mean_absolute_error(chlorine_actual, chlorine_predicted)
        ),
        "chlorine_rmse": float(
            np.sqrt(mean_squared_error(chlorine_actual, chlorine_predicted))
        ),
        "chlorine_r2": float(r2_score(chlorine_actual, chlorine_predicted)),
    }


def calculate_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate sensitivity (recall/TPR) and specificity (TNR).
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = np.sum(y_pred & y_true)
    tn = np.sum(~y_pred & ~y_true)
    fp = np.sum(y_pred & ~y_true)
    fn = np.sum(~y_pred & y_true)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return float(sensitivity), float(specificity)


def _get_fig_ax(ax=None, figsize=(8, 6)):
    """
    Helper for plotting functions.

    If ax is provided, plot into that axis.
    Otherwise, create a new figure and axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    return fig, ax


def plot_roc_curve_event_detection(
    result: DetectionResult,
    title: str = "ROC Curve - Event Detection",
    figsize: tuple = (8, 6),
    ax=None,
) -> tuple:
    """
    Plot ROC curve for event detection using event probabilities.

    Args:
        result: DetectionResult object
        title: Plot title
        figsize: Figure size, used only if ax is None
        ax: Optional matplotlib axis to plot into

    Returns:
        (fpr, tpr, roc_auc, figure, axes)
    """
    y_true = result.event_flags.astype(bool)
    y_scores = result.event_probability

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random Classifier",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    return fpr, tpr, roc_auc, fig, ax


def plot_roc_curve_chlorine_prediction(
    result: DetectionResult,
    chlorine_indices: list[int],
    threshold: float = 0.5,
    title: str = "ROC Curve - Chlorine Prediction",
    figsize: tuple = (8, 6),
    ax=None,
) -> tuple:
    """
    Plot contamination ROC using chlorine prediction residuals as anomaly scores.

    This does not measure chlorine regression accuracy. A strong predictor may
    also predict event-period chlorine well, producing an AUC near 0.5.

    Args:
        result: DetectionResult object
        chlorine_indices: Indices of chlorine channels in the output
        threshold: Residual threshold for anomaly detection.
                   Kept for API compatibility, but ROC uses continuous scores.
        title: Plot title
        figsize: Figure size, used only if ax is None
        ax: Optional matplotlib axis to plot into

    Returns:
        (fpr, tpr, roc_auc, figure, axes)
    """
    chlorine_predicted = result.predicted[:, chlorine_indices]
    chlorine_actual = result.actual[:, chlorine_indices]

    chlorine_residuals = np.abs(chlorine_predicted - chlorine_actual).mean(axis=1)

    y_true = result.event_flags.astype(bool)
    y_scores = chlorine_residuals

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    ax.plot(
        fpr,
        tpr,
        color="green",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random Classifier",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    return fpr, tpr, roc_auc, fig, ax


def plot_chlorine_prediction_quality(
    result: DetectionResult,
    chlorine_indices: list[int] | None = None,
    title: str = "Chlorine prediction quality",
    figsize: tuple = (8, 6),
    ax=None,
) -> tuple:
    """Plot predicted versus observed chlorine and report regression metrics."""
    if chlorine_indices is None:
        chlorine_indices = list(range(result.actual.shape[1]))
    actual = result.actual[:, chlorine_indices].reshape(-1)
    predicted = result.predicted[:, chlorine_indices].reshape(-1)
    summary = get_chlorine_regression_summary(result, chlorine_indices)
    r2 = summary["chlorine_r2"]
    rmse = summary["chlorine_rmse"]
    mae = summary["chlorine_mae"]

    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)
    ax.scatter(actual, predicted, s=8, alpha=0.25, color="teal")
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    ax.plot([lower, upper], [lower, upper], "--", color="black", lw=1.5,
            label="Perfect prediction")
    ax.set_xlabel("Observed chlorine")
    ax.set_ylabel("Predicted chlorine")
    ax.set_title(title)
    ax.text(
        0.03, 0.97,
        f"R² = {r2:.3f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return r2, rmse, mae, fig, ax


def plot_precision_recall_curve(
    result: DetectionResult,
    title: str = "Precision-Recall Curve",
    figsize: tuple = (8, 6),
    ax=None,
) -> tuple:
    """
    Plot precision-recall curve for event detection.

    Args:
        result: DetectionResult object
        title: Plot title
        figsize: Figure size, used only if ax is None
        ax: Optional matplotlib axis to plot into

    Returns:
        (precision, recall, f1_scores, figure, axes)
    """
    y_true = result.event_flags.astype(bool)
    y_scores = result.event_probability

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    f1_scores = (
        2
        * (precision[:-1] * recall[:-1])
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    ap = average_precision_score(y_true, y_scores)

    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    ax.plot(
        recall,
        precision,
        color="purple",
        lw=2,
        label=f"Precision-Recall (AP = {ap:.3f})",
    )
    ax.fill_between(recall, precision, alpha=0.2, color="purple")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    return precision, recall, f1_scores, fig, ax


def plot_evaluation_metrics(
    result: DetectionResult,
    chlorine_indices: list[int] = None,
) -> dict:
    """
    Create a comprehensive evaluation dashboard with multiple plots.

    Args:
        result: DetectionResult object
        chlorine_indices: Indices of chlorine channels (for chlorine evaluation)

    Returns:
        Dictionary with all metrics and figures
    """
    metrics = {}
    figures = {}

    sensitivity, specificity = calculate_sensitivity_specificity(
        result.event_flags,
        result.alarms,
    )
    metrics["event_detection_sensitivity"] = sensitivity
    metrics["event_detection_specificity"] = specificity

    fpr, tpr, roc_auc, fig, ax = plot_roc_curve_event_detection(result)
    figures["roc_event_detection"] = fig
    metrics["event_detection_auc"] = roc_auc

    precision, recall, f1_scores, fig, ax = plot_precision_recall_curve(result)
    figures["precision_recall"] = fig
    metrics["event_detection_average_precision"] = float(
        average_precision_score(
            result.event_flags.astype(bool),
            result.event_probability,
        )
    )

    if chlorine_indices is not None:
        fpr_cl, tpr_cl, roc_auc_cl, fig_cl, ax_cl = plot_roc_curve_chlorine_prediction(
            result,
            chlorine_indices,
        )
        figures["roc_chlorine"] = fig_cl
        metrics["chlorine_auc"] = roc_auc_cl

    return {
        "metrics": metrics,
        "figures": figures,
    }


def get_evaluation_summary(result: DetectionResult) -> dict[str, float]:
    """
    Get a comprehensive summary of evaluation metrics.

    Args:
        result: DetectionResult object

    Returns:
        Dictionary with all key metrics
    """
    y_true = result.event_flags.astype(bool)
    y_pred = result.alarms.astype(bool)
    y_scores = result.event_probability

    tp = np.sum(y_pred & y_true)
    tn = np.sum(~y_pred & ~y_true)
    fp = np.sum(y_pred & ~y_true)
    fn = np.sum(~y_pred & y_true)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)

    roc_auc = roc_auc_score(y_true, y_scores)

    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    }
