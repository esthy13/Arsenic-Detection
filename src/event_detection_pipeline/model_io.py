"""
Utilities for saving and loading trained models and configurations.
"""

import pickle
import torch
from pathlib import Path
from .classes import Scalers, ThresholdConfig, SensorGroups
from .model import EventClassificationANN, WaterQualityANN
from .pipeline import EventDetector


def _infer_hidden_sizes(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    linear_weight_keys = sorted(
        (
            key for key in state_dict
            if key.startswith("network.") and key.endswith(".weight")
        ),
        key=lambda key: int(key.split(".")[1]),
    )
    if not linear_weight_keys:
        raise ValueError("Could not infer model architecture from saved weights")
    return tuple(int(state_dict[key].shape[0]) for key in linear_weight_keys[:-1])


def save_detector(
    detector: EventDetector,
    directory: str = "weights",
) -> None:
    """
    Save detector model, scalers, threshold config, and metadata to disk.
    Creates the directory and all parent directories if they don't exist.
    
    Args:
        detector: EventDetector instance to save
        directory: Directory path to save files (will be created if it doesn't exist)
    """
    # Create directory and all parent directories if they don't exist
    save_dir = Path(directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model weights
    model_path = save_dir / "model_weights.pth"
    torch.save([model.state_dict() for model in detector.models], model_path)
    if detector.event_classifier is not None:
        torch.save(
            detector.event_classifier.state_dict(),
            save_dir / "event_classifier_weights.pth",
        )
    #print(f"Model weights saved to {model_path}")
    
    # Save the scalers (for preprocessing)
    scalers_path = save_dir / "scalers.pkl"
    with open(scalers_path, "wb") as f:
        pickle.dump(detector.target_scalers, f)
    #print(f"Scalers saved to {scalers_path}")
    
    # Save the threshold config
    threshold_path = save_dir / "threshold.pkl"
    with open(threshold_path, "wb") as f:
        pickle.dump(detector.thresholds, f)
    #print(f"Threshold config saved to {threshold_path}")
    
    # Save other metadata
    metadata = {
        "history": detector.history,
        "epochs": detector.epochs,
        "batch_size": detector.batch_size,
        "learning_rate": detector.learning_rate,
        "hidden_sizes": detector.hidden_sizes,
        "groups": detector.groups,
        "device": str(detector.device),
        "true_positive_rates": detector.true_positive_rates,
        "false_positive_rates": detector.false_positive_rates,
        "model_format": "per_target",
        "feature_format": "flows_target_history_daily_time_v2",
        "label_mode": detector.label_mode,
        "arsenic_threshold": detector.arsenic_threshold,
        "fusion_method": detector.fusion_method,
        "alarm_threshold": detector.alarm_threshold,
        "active_target_indices": detector.active_target_indices,
        "max_validation_false_alarm_rate": detector.max_validation_false_alarm_rate,
        "dropout": detector.dropout,
        "weight_decay": detector.weight_decay,
        "early_stopping_patience": detector.early_stopping_patience,
        "classifier_input_mean": detector.classifier_input_mean,
        "classifier_input_std": detector.classifier_input_std,
        "classifier_hidden_sizes": detector.classifier_hidden_sizes,
        "classifier_dropout": detector.classifier_dropout,
        "grasp_report": detector.grasp_report,
    }
    metadata_path = save_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    #print(f"Metadata saved to {metadata_path}")
    
    print(f"\nAll files saved to {save_dir.resolve()}/")


def load_detector(
    directory: str = "weights",
    device: str = "cpu",
) -> EventDetector:
    """
    Load detector model, scalers, threshold config, and metadata from disk.
    
    Args:
        directory: Directory path where files are saved
        device: Device to load model to ('cpu', 'cuda', 'mps')
        
    Returns:
        EventDetector instance
    """
    load_dir = Path(directory)
    
    if not load_dir.exists():
        raise FileNotFoundError(f"Directory {load_dir.resolve()} does not exist")
    
    # Load metadata
    metadata_path = load_dir / "metadata.pkl"
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    if metadata.get("feature_format") != "flows_target_history_daily_time_v2":
        raise ValueError(
            "This detector uses the previous feature layout. Retrain it before "
            "loading it with the new pipeline."
        )
    #print(f"Metadata loaded from {metadata_path}")
    
    # Load scalers
    scalers_path = load_dir / "scalers.pkl"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    #print(f"Scalers loaded from {scalers_path}")
    
    # Load threshold config
    threshold_path = load_dir / "threshold.pkl"
    with open(threshold_path, "rb") as f:
        thresholds = pickle.load(f)
    #print(f"Threshold config loaded from {threshold_path}")
    
    torch_device = torch.device(device)
    model_path = load_dir / "model_weights.pth"
    state_dicts = torch.load(model_path, map_location=torch_device)
    if isinstance(state_dicts, dict):
        raise ValueError("This saved detector uses the old single-model format. Retrain and save it with the per-target pipeline.")

    hidden_sizes = tuple(metadata["hidden_sizes"])
    models: list[WaterQualityANN] = []
    for scaler, state_dict in zip(scalers, state_dicts):
        model = WaterQualityANN(
            input_dim=scaler.input_mean.shape[0],
            output_dim=scaler.target_mean.shape[0],
            hidden_sizes=hidden_sizes,
            dropout=metadata.get("dropout", 0.1),
        )
        model.load_state_dict(state_dict)
        model.to(torch_device)
        models.append(model)
    event_classifier = None
    classifier_path = load_dir / "event_classifier_weights.pth"
    if classifier_path.exists():
        event_classifier = EventClassificationANN(
            input_dim=len(metadata["classifier_input_mean"]),
            hidden_sizes=metadata.get("classifier_hidden_sizes", (128, 64)),
            dropout=metadata.get("classifier_dropout", 0.1),
        ).to(torch_device)
        event_classifier.load_state_dict(
            torch.load(classifier_path, map_location=torch_device)
        )
    
    # Recreate EventDetector
    detector = EventDetector(
        models=models,
        scalers=scalers,
        thresholds=thresholds,
        true_positive_rates=metadata["true_positive_rates"],
        false_positive_rates=metadata["false_positive_rates"],
        device=torch_device,
        history=metadata["history"],
        groups=metadata["groups"],
        epochs=metadata.get("epochs", 80),
        batch_size=metadata.get("batch_size", 128),
        learning_rate=metadata.get("learning_rate", 1e-3),
        hidden_sizes=hidden_sizes,
        label_mode=metadata.get("label_mode", "arsenic_arrival"),
        arsenic_threshold=metadata.get("arsenic_threshold", 0.01),
        fusion_method=metadata.get("fusion_method", "mean"),
        alarm_threshold=metadata.get("alarm_threshold", 0.5),
        active_target_indices=metadata.get("active_target_indices"),
        max_validation_false_alarm_rate=metadata.get(
            "max_validation_false_alarm_rate", 0.1
        ),
        dropout=metadata.get("dropout", 0.1),
        weight_decay=metadata.get("weight_decay", 1e-4),
        early_stopping_patience=metadata.get("early_stopping_patience", 12),
        event_classifier=event_classifier,
        classifier_input_mean=metadata.get("classifier_input_mean"),
        classifier_input_std=metadata.get("classifier_input_std"),
        classifier_hidden_sizes=metadata.get("classifier_hidden_sizes", (128, 64)),
        classifier_dropout=metadata.get("classifier_dropout", 0.1),
        grasp_report=metadata.get("grasp_report"),
    )
    
    print(f"\nDetector successfully loaded from {load_dir.resolve()}/")
    return detector
