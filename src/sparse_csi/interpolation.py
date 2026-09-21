"""Interpolation baselines for complex multi-subcarrier CSI fields."""

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree


def nearest_reconstruction(
    h_true: np.ndarray,
    coordinates: np.ndarray,
    sample_mask: np.ndarray,
) -> np.ndarray:
    """Copy every subcarrier from the nearest measured spatial position."""
    sample_points = coordinates[sample_mask.ravel()]
    tree = cKDTree(sample_points)
    _, nearest_sample_indices = tree.query(coordinates, k=1)

    sample_values = h_true[:, sample_mask]
    reconstructed_flat = sample_values[:, nearest_sample_indices]
    reconstruction = reconstructed_flat.reshape(h_true.shape).astype(np.complex64)
    reconstruction[:, sample_mask] = h_true[:, sample_mask]
    return reconstruction


def linear_reconstruction(
    h_true: np.ndarray,
    coordinates: np.ndarray,
    sample_mask: np.ndarray,
    nearest_result: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate real and imaginary channels with one shared triangulation."""
    sample_points = coordinates[sample_mask.ravel()]
    sample_values = h_true[:, sample_mask]
    subcarrier_count = h_true.shape[0]

    vector_values = np.concatenate(
        (sample_values.real.T, sample_values.imag.T),
        axis=1,
    )
    interpolator = LinearNDInterpolator(
        Delaunay(sample_points),
        vector_values,
        fill_value=np.nan,
    )
    interpolated = np.asarray(interpolator(coordinates))

    missing_rows = np.isnan(interpolated).any(axis=1)
    if np.any(missing_rows):
        if nearest_result is None:
            nearest_result = nearest_reconstruction(
                h_true,
                coordinates,
                sample_mask,
            )
        nearest_flat = nearest_result.reshape(subcarrier_count, -1).T
        nearest_vectors = np.concatenate(
            (nearest_flat.real, nearest_flat.imag),
            axis=1,
        )
        interpolated[missing_rows] = nearest_vectors[missing_rows]

    reconstruction_flat = (
        interpolated[:, :subcarrier_count]
        + 1j * interpolated[:, subcarrier_count:]
    ).T
    reconstruction = reconstruction_flat.reshape(h_true.shape).astype(np.complex64)
    reconstruction[:, sample_mask] = h_true[:, sample_mask]
    return reconstruction
