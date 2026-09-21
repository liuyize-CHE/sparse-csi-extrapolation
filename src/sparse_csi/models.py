"""Fourier-feature MLP and its reproducible fitting procedure."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    max_frequency_cpm: float
    frequency_band_count: int = 6
    direction_count: int = 8
    hidden_width: int = 128
    hidden_depth: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    max_epochs: int = 2500
    patience: int = 200
    validation_fraction: float = 0.20


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def run_seed_for(scene_index: int, rate_index: int, repeat: int, seed: int) -> int:
    """Derive a stable seed for one scene/rate/repeat combination."""
    return seed + scene_index * 10_000 + rate_index * 100 + repeat


class FourierEncoding(nn.Module):
    """Deterministic plane-wave features with frequencies in cycles/metre."""

    def __init__(
        self,
        max_frequency_cpm: float,
        frequency_band_count: int,
        direction_count: int,
    ) -> None:
        super().__init__()
        if max_frequency_cpm <= 0:
            raise ValueError("max_frequency_cpm must be positive")
        if frequency_band_count < 1 or direction_count < 1:
            raise ValueError("Fourier band and direction counts must be positive")

        frequencies = torch.linspace(
            max_frequency_cpm / frequency_band_count,
            max_frequency_cpm,
            frequency_band_count,
            dtype=torch.float32,
        )
        angles = torch.arange(direction_count, dtype=torch.float32)
        angles = angles * (math.pi / direction_count)
        directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        projection_matrix = (frequencies[:, None, None] * directions[None]).reshape(
            -1, 2
        )
        self.register_buffer("projection_matrix", projection_matrix)
        self.output_dimension = 2 + 2 * projection_matrix.shape[0]

    def forward(self, coordinates_m: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * math.pi * coordinates_m @ self.projection_matrix.T
        raw_coordinates = coordinates_m / 2.0 - 1.0
        return torch.cat(
            (raw_coordinates, torch.sin(projected), torch.cos(projected)),
            dim=-1,
        )


class FourierMLP(nn.Module):
    def __init__(self, output_dimension: int, config: ModelConfig) -> None:
        super().__init__()
        self.encoding = FourierEncoding(
            config.max_frequency_cpm,
            config.frequency_band_count,
            config.direction_count,
        )
        layers: list[nn.Module] = []
        input_dimension = self.encoding.output_dimension
        for _ in range(config.hidden_depth):
            layers.extend(
                (
                    nn.Linear(input_dimension, config.hidden_width),
                    nn.ReLU(),
                )
            )
            input_dimension = config.hidden_width
        layers.append(nn.Linear(input_dimension, output_dimension))
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates_m: torch.Tensor) -> torch.Tensor:
        return self.network(self.encoding(coordinates_m))


def complex_to_targets(values: np.ndarray, scale: float) -> np.ndarray:
    """Convert complex CSI to normalized real/imaginary regression targets."""
    return np.concatenate((values.real.T, values.imag.T), axis=1).astype(
        np.float32
    ) / scale


def targets_to_complex(
    targets: np.ndarray,
    subcarrier_count: int,
    scale: float,
) -> np.ndarray:
    real = targets[:, :subcarrier_count]
    imaginary = targets[:, subcarrier_count:]
    return ((real + 1j * imaginary) * scale).T.astype(np.complex64)


def observed_scale(h_true: np.ndarray, sample_mask: np.ndarray) -> float:
    values = h_true[:, sample_mask]
    return max(float(np.sqrt(np.mean(np.abs(values) ** 2))), 1.0e-6)


def split_measured_positions(
    sample_mask: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    measured_indices = np.flatnonzero(sample_mask.ravel())
    if measured_indices.size < 10:
        raise ValueError("Too few measured positions for an internal split.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(measured_indices)
    validation_count = max(1, int(round(validation_fraction * shuffled.size)))
    return shuffled[validation_count:], shuffled[:validation_count]


def make_tensors(
    coordinates: np.ndarray,
    h_true: np.ndarray,
    flat_indices: np.ndarray,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_values = h_true.reshape(h_true.shape[0], -1)[:, flat_indices]
    coordinate_tensor = torch.as_tensor(
        coordinates[flat_indices], dtype=torch.float32, device=device
    )
    target_tensor = torch.as_tensor(
        complex_to_targets(flat_values, scale),
        dtype=torch.float32,
        device=device,
    )
    return coordinate_tensor, target_tensor


def train_with_early_stopping(
    h_true: np.ndarray,
    coordinates: np.ndarray,
    sample_mask: np.ndarray,
    config: ModelConfig,
    seed: int,
    device: torch.device,
) -> tuple[int, float, float]:
    training_indices, validation_indices = split_measured_positions(
        sample_mask,
        config.validation_fraction,
        seed + 17,
    )
    scale = observed_scale(h_true, sample_mask)
    train_x, train_y = make_tensors(
        coordinates, h_true, training_indices, scale, device
    )
    validation_x, validation_y = make_tensors(
        coordinates, h_true, validation_indices, scale, device
    )

    set_reproducible_seed(seed)
    model = FourierMLP(train_y.shape[1], config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = math.inf
    best_training_loss = math.inf
    best_epoch = 1
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_loss = loss_function(model(train_x), train_y)
        training_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x), validation_y))
        if validation_loss < best_validation_loss - 1.0e-7:
            best_validation_loss = validation_loss
            best_training_loss = float(training_loss.detach())
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    return best_epoch, best_training_loss, best_validation_loss


def fit_all_measurements(
    h_true: np.ndarray,
    coordinates: np.ndarray,
    sample_mask: np.ndarray,
    config: ModelConfig,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    measured_indices = np.flatnonzero(sample_mask.ravel())
    scale = observed_scale(h_true, sample_mask)
    train_x, train_y = make_tensors(
        coordinates, h_true, measured_indices, scale, device
    )

    set_reproducible_seed(seed)
    model = FourierMLP(train_y.shape[1], config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()

    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_loss = loss_function(model(train_x), train_y)
        training_loss.backward()
        optimizer.step()

    model.eval()
    predictions = []
    coordinate_tensor = torch.as_tensor(coordinates, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(coordinates), 4096):
            batch = coordinate_tensor[start : start + 4096].to(device)
            predictions.append(model(batch).cpu().numpy())

    predicted_targets = np.concatenate(predictions, axis=0)
    predicted_flat = targets_to_complex(
        predicted_targets,
        h_true.shape[0],
        scale,
    )
    reconstruction = predicted_flat.reshape(h_true.shape)
    reconstruction[:, sample_mask] = h_true[:, sample_mask]
    return reconstruction, float(training_loss.detach().cpu())


def fit_and_reconstruct(
    h_true: np.ndarray,
    coordinates: np.ndarray,
    sample_mask: np.ndarray,
    config: ModelConfig,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float | int]]:
    best_epoch, internal_train_loss, internal_validation_loss = (
        train_with_early_stopping(
            h_true,
            coordinates,
            sample_mask,
            config,
            seed,
            device,
        )
    )
    reconstruction, full_training_loss = fit_all_measurements(
        h_true,
        coordinates,
        sample_mask,
        config,
        best_epoch,
        seed,
        device,
    )
    diagnostics: dict[str, float | int] = {
        "selected_epoch": best_epoch,
        "internal_train_loss": internal_train_loss,
        "internal_validation_loss": internal_validation_loss,
        "full_training_loss": full_training_loss,
    }
    return reconstruction, diagnostics
