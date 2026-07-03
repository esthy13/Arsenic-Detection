"""
Utilities for saving and loading trained models and configurations.
"""

import pickle
import torch
from pathlib import Path
from .classes import Scalers, ThresholdConfig, SensorGroups
from .model import WaterQualityANN
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
    torch.save(detector.model.state_dict(), model_path)
    #print(f"Model weights saved to {model_path}")
    
    # Save the scalers (for preprocessing)
    scalers_path = save_dir / "scalers.pkl"
    with open(scalers_path, "wb") as f:
        pickle.dump(detector.scalers, f)
    #print(f"Scalers saved to {scalers_path}")
    
    # Save the threshold config
    threshold_path = save_dir / "threshold.pkl"
    with open(threshold_path, "wb") as f:
        pickle.dump(detector.threshold, f)
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
        "true_positive_rate": detector.true_positive_rate,
        "false_positive_rate": detector.false_positive_rate,
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
    #print(f"Metadata loaded from {metadata_path}")
    
    # Load scalers
    scalers_path = load_dir / "scalers.pkl"
    with open(scalers_path, "rb") as f:
        scalers: Scalers = pickle.load(f)
    #print(f"Scalers loaded from {scalers_path}")
    
    # Load threshold config
    threshold_path = load_dir / "threshold.pkl"
    with open(threshold_path, "rb") as f:
        threshold: ThresholdConfig = pickle.load(f)
    #print(f"Threshold config loaded from {threshold_path}")
    
    torch_device = torch.device(device)
    input_dim = scalers.input_mean.shape[0]
    output_dim = scalers.target_mean.shape[0]
    model_path = load_dir / "model_weights.pth"
    state_dict = torch.load(model_path, map_location=torch_device)
    hidden_sizes = tuple(metadata["hidden_sizes"]) if "hidden_sizes" in metadata else _infer_hidden_sizes(state_dict)
    
    # Recreate model
    model = WaterQualityANN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
    )
    
    # Load model weights
    model.load_state_dict(state_dict)
    model.to(torch_device)
    #print(f"Model weights loaded from {model_path}")
    
    # Recreate EventDetector
    detector = EventDetector(
        model=model,
        scalers=scalers,
        threshold=threshold,
        true_positive_rate=metadata["true_positive_rate"],
        false_positive_rate=metadata["false_positive_rate"],
        device=torch_device,
        history=metadata["history"],
        groups=metadata["groups"],
        epochs=metadata.get("epochs", 80),
        batch_size=metadata.get("batch_size", 128),
        learning_rate=metadata.get("learning_rate", 1e-3),
        hidden_sizes=hidden_sizes,
    )
    
    print(f"\nDetector successfully loaded from {load_dir.resolve()}/")
    return detector
