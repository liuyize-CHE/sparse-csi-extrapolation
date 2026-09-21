"""Dataset loading, validation, and spatial sampling helpers."""

from pathlib import Path

import numpy as np


REQUIRED_DATASET_KEYS = {
    "csi",
    "x_m",
    "y_m",
    "frequencies_hz",
    "valid_masks",
    "validation_scene_indices",
    "test_scene_indices",
}


def load_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    """Load a generated CSI dataset and validate its core arrays."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run experiments/generate_dataset.py first."
        )

    with np.load(dataset_path) as loaded:
        dataset = {key: loaded[key] for key in loaded.files}

    missing_keys = REQUIRED_DATASET_KEYS.difference(dataset)
    if missing_keys:
        raise KeyError(f"Dataset is missing arrays: {sorted(missing_keys)}")

    csi = dataset["csi"]
    if csi.ndim != 4 or not np.iscomplexobj(csi):
        raise ValueError("Expected complex CSI with shape (scene, subcarrier, y, x).")
    if not np.isfinite(csi).all():
        raise ValueError("CSI contains NaN or infinite values.")

    expected_mask_shape = (csi.shape[0], csi.shape[2], csi.shape[3])
    if dataset["valid_masks"].shape != expected_mask_shape:
        raise ValueError("valid_masks shape does not match the CSI tensor.")
    return dataset


def build_coordinates(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Return flattened ``(x, y)`` coordinates for a rectangular grid."""
    xx, yy = np.meshgrid(x_m, y_m)
    return np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)


def draw_random_spatial_mask(
    valid_mask: np.ndarray,
    sampling_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a reproducible random subset of valid spatial positions."""
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError("sampling_rate must be in (0, 1].")

    candidate_indices = np.flatnonzero(valid_mask)
    sample_count = max(3, int(round(sampling_rate * candidate_indices.size)))
    selected_indices = rng.choice(candidate_indices, size=sample_count, replace=False)
    sample_mask = np.zeros_like(valid_mask, dtype=bool)
    sample_mask.flat[selected_indices] = True
    return sample_mask
