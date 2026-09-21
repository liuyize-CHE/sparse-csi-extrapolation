"""Shared project paths and experiment constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "initial_csi_dataset.npz"
SAMPLING_RATES = (0.05, 0.10, 0.20)
SPEED_OF_LIGHT_MPS = 299_792_458.0
