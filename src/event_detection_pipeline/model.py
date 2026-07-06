from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .classes import Scalers


class WaterQualityANN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int] = (128, 64)) -> None:
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_size))
            #Gaussian Error Linear Unit
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_scalers(train_inputs: np.ndarray, train_targets: np.ndarray) -> Scalers:
    input_mean = train_inputs.mean(axis=0)
    input_std = train_inputs.std(axis=0)
    target_mean = train_targets.mean(axis=0)
    target_std = train_targets.std(axis=0)

    input_std[input_std == 0] = 1.0
    target_std[target_std == 0] = 1.0

    return Scalers(
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def apply_scalers(inputs: np.ndarray, targets: np.ndarray, scalers: Scalers) -> tuple[np.ndarray, np.ndarray]:
    scaled_inputs = (inputs - scalers.input_mean) / scalers.input_std
    scaled_targets = (targets - scalers.target_mean) / scalers.target_std
    return scaled_inputs, scaled_targets


def inverse_targets(values: np.ndarray, scalers: Scalers) -> np.ndarray:
    return values * scalers.target_std + scalers.target_mean


def train_model(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    device: torch.device,
    hidden_sizes: Sequence[int] = (128, 64),
    batch_size: int = 128,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> WaterQualityANN:
    model = WaterQualityANN(train_inputs.shape[1], train_targets.shape[1], hidden_sizes=hidden_sizes).to(device)

    dataset = TensorDataset(
        torch.as_tensor(train_inputs, dtype=torch.float32),
        torch.as_tensor(train_targets, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

    return model


def predict(model: WaterQualityANN, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor_inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        tensor_predictions = model(tensor_inputs).detach().cpu().numpy()
    return tensor_predictions


def residual_series(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.mean(actual - predicted, axis=1)
