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
    f1_score
)
from .classes import DetectionResult


def calculate_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate sensitivity (recall/TPR) and specificity (TNR).
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        
    Returns:
        (sensitivity, specificity)
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


def plot_roc_curve_event_detection(
    result: DetectionResult,
    title: str = "ROC Curve - Event Detection",
    figsize: tuple = (8, 6),
) -> tuple:
    """
    Plot ROC curve for event detection using event probabilities.
    
    Args:
        result: DetectionResult object
        title: Plot title
        figsize: Figure size
        
    Returns:
        (fpr, tpr, roc_auc, figure, axes)
    """
    y_true = result.event_flags.astype(bool)
    y_scores = result.event_probability
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
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
) -> tuple:
    """
    Plot ROC curve for chlorine prediction by comparing predicted vs actual.
    Converts residuals to binary predictions using a threshold.
    
    Args:
        result: DetectionResult object
        chlorine_indices: Indices of chlorine channels in the output
        threshold: Residual threshold for anomaly detection
        title: Plot title
        figsize: Figure size
        
    Returns:
        (fpr, tpr, roc_auc, figure, axes)
    """
    # Extract chlorine predictions and actuals
    chlorine_predicted = result.predicted[:, chlorine_indices]
    chlorine_actual = result.actual[:, chlorine_indices]
    
    # Calculate residuals for chlorine
    chlorine_residuals = np.abs(chlorine_predicted - chlorine_actual).mean(axis=1)
    
    # Use event flags as ground truth (assuming events cause chlorine anomalies)
    y_true = result.event_flags.astype(bool)
    y_scores = chlorine_residuals
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    return fpr, tpr, roc_auc, fig, ax


def plot_precision_recall_curve(
    result: DetectionResult,
    title: str = "Precision-Recall Curve",
    figsize: tuple = (8, 6),
) -> tuple:
    """
    Plot precision-recall curve for event detection.
    
    Args:
        result: DetectionResult object
        title: Plot title
        figsize: Figure size
        
    Returns:
        (precision, recall, f1_scores, figure, axes)
    """
    y_true = result.event_flags.astype(bool)
    y_scores = result.event_probability
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    # Average precision
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color='purple', lw=2, label=f'Precision-Recall (AP = {ap:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='purple')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
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
    
    # Event detection metrics
    sensitivity, specificity = calculate_sensitivity_specificity(
        result.event_flags,
        result.alarms
    )
    metrics['event_detection_sensitivity'] = sensitivity
    metrics['event_detection_specificity'] = specificity
    
    # ROC curve for event detection
    fpr, tpr, roc_auc, fig, ax = plot_roc_curve_event_detection(result)
    figures['roc_event_detection'] = fig
    metrics['event_detection_auc'] = roc_auc
    
    # Precision-recall curve
    precision, recall, f1_scores, fig, ax = plot_precision_recall_curve(result)
    figures['precision_recall'] = fig
    metrics['event_detection_average_precision'] = float(np.mean(precision[:-1]))
    
    # Chlorine prediction metrics (if indices provided)
    if chlorine_indices is not None:
        fpr_cl, tpr_cl, roc_auc_cl, fig_cl, ax_cl = plot_roc_curve_chlorine_prediction(
            result,
            chlorine_indices
        )
        figures['roc_chlorine'] = fig_cl
        metrics['chlorine_auc'] = roc_auc_cl
    
    return {
        'metrics': metrics,
        'figures': figures,
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
    
    # Confusion matrix metrics
    tp = np.sum(y_pred & y_true)
    tn = np.sum(~y_pred & ~y_true)
    fp = np.sum(y_pred & ~y_true)
    fn = np.sum(~y_pred & y_true)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    
    # ROC metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    
    return {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
    }
