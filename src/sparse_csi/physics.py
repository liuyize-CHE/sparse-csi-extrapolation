"""Propagation-informed transforms used by the Tx-aware model."""

import numpy as np

from sparse_csi.paths import SPEED_OF_LIGHT_MPS


def direct_phase_factor(
    coordinates: np.ndarray,
    tx_position_m: np.ndarray,
    frequencies_hz: np.ndarray,
    field_shape: tuple[int, int, int],
) -> np.ndarray:
    """Return the conjugate direct-path carrier phase at every grid point."""
    distances_m = np.linalg.norm(coordinates - tx_position_m[None, :], axis=1)
    phase = (
        2.0
        * np.pi
        * frequencies_hz[:, None]
        * distances_m[None, :]
        / SPEED_OF_LIGHT_MPS
    )
    return np.exp(1j * phase).reshape(field_shape).astype(np.complex64)


def compensate_direct_phase(
    h_true: np.ndarray,
    phase_factor: np.ndarray,
) -> np.ndarray:
    return (h_true * phase_factor).astype(np.complex64)


def restore_direct_phase(
    compensated_prediction: np.ndarray,
    phase_factor: np.ndarray,
) -> np.ndarray:
    return (compensated_prediction * np.conj(phase_factor)).astype(np.complex64)
