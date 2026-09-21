"""Generate a reproducible multi-scene, multipath, multi-subcarrier CSI dataset.

The dataset is intentionally small enough for a course project while keeping
the three dimensions that matter for Task D:

* multiple propagation scenes;
* one direct path plus several reflected paths;
* complex CSI on multiple OFDM-like subcarriers at every spatial position.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPEED_OF_LIGHT = 3.0e8
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
FIGURE_DIR = DATA_ROOT / "figures"


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 2026
    num_scenes: int = 10
    num_subcarriers: int = 8
    num_reflectors: int = 3
    grid_size: int = 101
    region_size_m: float = 4.0
    center_frequency_hz: float = 3.0e9
    bandwidth_hz: float = 20.0e6
    epsilon_m: float = 0.05
    minimum_valid_distance_m: float = 0.15
    tx_margin_m: float = 0.60
    scatterer_margin_m: float = 0.20
    minimum_tx_scatterer_distance_m: float = 0.50
    minimum_scatterer_separation_m: float = 0.30
    path_loss_exponent_min: float = 2.6
    path_loss_exponent_max: float = 3.4
    reflection_magnitude_min: float = 0.25
    reflection_magnitude_max: float = 0.65


def sample_scatterers(
    rng: np.random.Generator,
    tx_position: np.ndarray,
    config: DatasetConfig,
) -> np.ndarray:
    """Sample separated scatterers that are not too close to the transmitter."""
    scatterers: list[np.ndarray] = []
    lower = config.scatterer_margin_m
    upper = config.region_size_m - config.scatterer_margin_m

    for _ in range(config.num_reflectors):
        for _attempt in range(10_000):
            candidate = rng.uniform(lower, upper, size=2)
            far_from_tx = (
                np.linalg.norm(candidate - tx_position)
                >= config.minimum_tx_scatterer_distance_m
            )
            separated = all(
                np.linalg.norm(candidate - existing)
                >= config.minimum_scatterer_separation_m
                for existing in scatterers
            )
            if far_from_tx and separated:
                scatterers.append(candidate)
                break
        else:
            raise RuntimeError("Could not sample a valid scatterer configuration.")

    return np.stack(scatterers, axis=0)


def generate_scene(
    rng: np.random.Generator,
    xx: np.ndarray,
    yy: np.ndarray,
    frequencies_hz: np.ndarray,
    config: DatasetConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
]:
    """Generate one scene with one direct and several reflected paths."""
    tx_position = rng.uniform(
        config.tx_margin_m,
        config.region_size_m - config.tx_margin_m,
        size=2,
    )
    scatterer_positions = sample_scatterers(rng, tx_position, config)
    path_loss_exponent = float(
        rng.uniform(
            config.path_loss_exponent_min,
            config.path_loss_exponent_max,
        )
    )

    reflection_magnitudes = rng.uniform(
        config.reflection_magnitude_min,
        config.reflection_magnitude_max,
        size=config.num_reflectors,
    )
    reflection_phase_shifts = rng.uniform(
        -np.pi,
        np.pi,
        size=config.num_reflectors,
    )
    reflection_coefficients = reflection_magnitudes * np.exp(
        1j * reflection_phase_shifts
    )

    direct_distance = np.hypot(xx - tx_position[0], yy - tx_position[1])
    direct_amplitude = 1.0 / (
        direct_distance + config.epsilon_m
    ) ** (path_loss_exponent / 2.0)
    direct_phase = (
        -2.0
        * np.pi
        * frequencies_hz[:, None, None]
        * direct_distance[None, :, :]
        / SPEED_OF_LIGHT
    )
    csi = direct_amplitude[None, :, :] * np.exp(1j * direct_phase)

    for scatterer, coefficient in zip(
        scatterer_positions,
        reflection_coefficients,
    ):
        tx_to_scatterer = np.linalg.norm(tx_position - scatterer)
        scatterer_to_rx = np.hypot(xx - scatterer[0], yy - scatterer[1])
        reflected_distance = tx_to_scatterer + scatterer_to_rx
        reflected_amplitude = coefficient / (
            reflected_distance + config.epsilon_m
        ) ** (path_loss_exponent / 2.0)
        reflected_phase = (
            -2.0
            * np.pi
            * frequencies_hz[:, None, None]
            * reflected_distance[None, :, :]
            / SPEED_OF_LIGHT
        )
        csi = csi + reflected_amplitude[None, :, :] * np.exp(
            1j * reflected_phase
        )

    valid_mask = direct_distance >= config.minimum_valid_distance_m
    return (
        csi.astype(np.complex64),
        tx_position.astype(np.float32),
        scatterer_positions.astype(np.float32),
        reflection_coefficients.astype(np.complex64),
        path_loss_exponent,
        valid_mask,
    )


def generate_dataset(config: DatasetConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    x_m = np.linspace(0.0, config.region_size_m, config.grid_size).astype(
        np.float32
    )
    y_m = np.linspace(0.0, config.region_size_m, config.grid_size).astype(
        np.float32
    )
    xx, yy = np.meshgrid(x_m, y_m)
    frequencies_hz = np.linspace(
        config.center_frequency_hz - config.bandwidth_hz / 2.0,
        config.center_frequency_hz + config.bandwidth_hz / 2.0,
        config.num_subcarriers,
    ).astype(np.float64)

    csi = np.empty(
        (
            config.num_scenes,
            config.num_subcarriers,
            config.grid_size,
            config.grid_size,
        ),
        dtype=np.complex64,
    )
    tx_positions_m = np.empty((config.num_scenes, 2), dtype=np.float32)
    scatterer_positions_m = np.empty(
        (config.num_scenes, config.num_reflectors, 2),
        dtype=np.float32,
    )
    reflection_coefficients = np.empty(
        (config.num_scenes, config.num_reflectors),
        dtype=np.complex64,
    )
    path_loss_exponents = np.empty(config.num_scenes, dtype=np.float32)
    valid_masks = np.empty(
        (config.num_scenes, config.grid_size, config.grid_size),
        dtype=bool,
    )

    for scene_index in range(config.num_scenes):
        (
            scene_csi,
            tx_position,
            scatterer_positions,
            scene_reflection_coefficients,
            path_loss_exponent,
            valid_mask,
        ) = generate_scene(rng, xx, yy, frequencies_hz, config)
        csi[scene_index] = scene_csi
        tx_positions_m[scene_index] = tx_position
        scatterer_positions_m[scene_index] = scatterer_positions
        reflection_coefficients[scene_index] = scene_reflection_coefficients
        path_loss_exponents[scene_index] = path_loss_exponent
        valid_masks[scene_index] = valid_mask

    scene_indices = np.arange(config.num_scenes, dtype=np.int64)
    split_rng = np.random.default_rng(config.seed + 1)
    split_rng.shuffle(scene_indices)
    train_end = int(round(0.6 * config.num_scenes))
    validation_end = train_end + int(round(0.2 * config.num_scenes))

    return {
        "csi": csi,
        "x_m": x_m,
        "y_m": y_m,
        "frequencies_hz": frequencies_hz,
        "tx_positions_m": tx_positions_m,
        "scatterer_positions_m": scatterer_positions_m,
        "reflection_coefficients": reflection_coefficients,
        "path_loss_exponents": path_loss_exponents,
        "valid_masks": valid_masks,
        "train_scene_indices": np.sort(scene_indices[:train_end]),
        "validation_scene_indices": np.sort(
            scene_indices[train_end:validation_end]
        ),
        "test_scene_indices": np.sort(scene_indices[validation_end:]),
    }


def validate_dataset(
    dataset: dict[str, np.ndarray],
    config: DatasetConfig,
) -> dict[str, float | int | str | bool | list[int]]:
    expected_shape = (
        config.num_scenes,
        config.num_subcarriers,
        config.grid_size,
        config.grid_size,
    )
    csi = dataset["csi"]
    if csi.shape != expected_shape:
        raise AssertionError(f"Unexpected CSI shape: {csi.shape} != {expected_shape}")
    if not np.iscomplexobj(csi):
        raise AssertionError("CSI must be stored as a complex-valued array.")
    if not np.isfinite(csi).all():
        raise AssertionError("CSI contains NaN or infinite values.")

    grid_spacing_m = config.region_size_m / (config.grid_size - 1)
    shortest_wavelength_m = SPEED_OF_LIGHT / float(
        np.max(dataset["frequencies_hz"])
    )
    nyquist_spacing_m = shortest_wavelength_m / 2.0
    if grid_spacing_m > nyquist_spacing_m:
        raise AssertionError(
            "Spatial grid is too coarse for the highest carrier frequency: "
            f"spacing={grid_spacing_m:.6f} m, "
            f"half-wavelength={nyquist_spacing_m:.6f} m."
        )

    split_arrays = (
        dataset["train_scene_indices"],
        dataset["validation_scene_indices"],
        dataset["test_scene_indices"],
    )
    combined_indices = np.concatenate(split_arrays)
    if not np.array_equal(np.sort(combined_indices), np.arange(config.num_scenes)):
        raise AssertionError("Scene splits are not disjoint and complete.")

    scene_differences = [
        float(np.mean(np.abs(csi[index] - csi[0])))
        for index in range(1, config.num_scenes)
    ]
    if min(scene_differences) <= 1.0e-6:
        raise AssertionError("At least two generated scenes are effectively identical.")

    adjacent_subcarrier_difference = float(
        np.mean(np.abs(csi[:, 1:] - csi[:, :-1]))
    )
    if adjacent_subcarrier_difference <= 1.0e-6:
        raise AssertionError("Subcarriers do not contain distinct CSI values.")

    valid_values = csi[
        np.broadcast_to(dataset["valid_masks"][:, None, :, :], csi.shape)
    ]
    return {
        "csi_shape": list(csi.shape),
        "csi_dtype": str(csi.dtype),
        "all_finite": bool(np.isfinite(csi).all()),
        "grid_spacing_m": grid_spacing_m,
        "shortest_wavelength_m": shortest_wavelength_m,
        "spatial_nyquist_spacing_m": nyquist_spacing_m,
        "spatial_nyquist_check_passed": True,
        "mean_amplitude": float(np.mean(np.abs(valid_values))),
        "median_amplitude": float(np.median(np.abs(valid_values))),
        "maximum_amplitude": float(np.max(np.abs(valid_values))),
        "mean_adjacent_subcarrier_difference": adjacent_subcarrier_difference,
        "minimum_mean_scene_difference_from_scene0": float(min(scene_differences)),
        "train_scenes": dataset["train_scene_indices"].tolist(),
        "validation_scenes": dataset["validation_scene_indices"].tolist(),
        "test_scenes": dataset["test_scene_indices"].tolist(),
    }


def save_dataset(
    dataset: dict[str, np.ndarray],
    config: DatasetConfig,
) -> tuple[Path, Path, dict[str, object]]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = PROCESSED_DIR / "initial_csi_dataset.npz"
    np.savez_compressed(dataset_path, **dataset)

    # Reload the serialized file so validation checks the actual artifact.
    with np.load(dataset_path) as loaded:
        reloaded = {key: loaded[key] for key in loaded.files}
    summary: dict[str, object] = validate_dataset(reloaded, config)
    summary["config"] = asdict(config)
    summary["dataset_file"] = dataset_path.name
    summary["dataset_size_bytes"] = dataset_path.stat().st_size
    summary["sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    metadata_path = PROCESSED_DIR / "initial_csi_dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dataset_path, metadata_path, summary


def save_overview_figure(
    dataset: dict[str, np.ndarray],
    config: DatasetConfig,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scene_index = 0
    subcarrier_index = config.num_subcarriers // 2
    field = dataset["csi"][scene_index, subcarrier_index]
    amplitude_db = 20.0 * np.log10(
        np.abs(field) / max(float(np.max(np.abs(field))), 1.0e-12) + 1.0e-12
    )
    phase = np.angle(field)
    extent = [0.0, config.region_size_m, 0.0, config.region_size_m]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    amplitude_image = axes[0].imshow(
        amplitude_db,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=-70,
        vmax=0,
    )
    axes[0].set_title("Scene 0: relative amplitude (dB)")
    fig.colorbar(amplitude_image, ax=axes[0])

    phase_image = axes[1].imshow(
        phase,
        origin="lower",
        extent=extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1].set_title("Scene 0: phase (rad)")
    fig.colorbar(phase_image, ax=axes[1])

    tx_position = dataset["tx_positions_m"][scene_index]
    scatterers = dataset["scatterer_positions_m"][scene_index]
    axes[2].scatter(
        scatterers[:, 0],
        scatterers[:, 1],
        marker="D",
        s=80,
        c="cyan",
        edgecolors="black",
        label="Reflectors",
    )
    axes[2].scatter(
        tx_position[0],
        tx_position[1],
        marker="*",
        s=220,
        c="red",
        edgecolors="black",
        label="Tx",
    )
    axes[2].set_title("Scene geometry")
    axes[2].set_xlim(0.0, config.region_size_m)
    axes[2].set_ylim(0.0, config.region_size_m)
    axes[2].set_aspect("equal")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

    frequency_ghz = dataset["frequencies_hz"][subcarrier_index] / 1.0e9
    fig.suptitle(
        f"Initial CSI dataset example: scene={scene_index}, "
        f"subcarrier={subcarrier_index}, f={frequency_ghz:.6f} GHz"
    )
    output_path = FIGURE_DIR / "initial_dataset_scene0_overview.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def save_subcarrier_figure(
    dataset: dict[str, np.ndarray],
    config: DatasetConfig,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    scene_index = 0
    selected_subcarriers = np.linspace(
        0,
        config.num_subcarriers - 1,
        4,
        dtype=int,
    )
    reference = float(np.max(np.abs(dataset["csi"][scene_index])))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    last_image = None
    for axis, subcarrier_index in zip(axes, selected_subcarriers):
        field = dataset["csi"][scene_index, subcarrier_index]
        amplitude_db = 20.0 * np.log10(
            np.abs(field) / max(reference, 1.0e-12) + 1.0e-12
        )
        last_image = axis.imshow(
            amplitude_db,
            origin="lower",
            extent=[0.0, config.region_size_m, 0.0, config.region_size_m],
            cmap="viridis",
            vmin=-70,
            vmax=0,
        )
        frequency_ghz = dataset["frequencies_hz"][subcarrier_index] / 1.0e9
        axis.set_title(f"k={subcarrier_index}, {frequency_ghz:.4f} GHz")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

    if last_image is not None:
        fig.colorbar(last_image, ax=axes, label="Relative amplitude (dB)")

    output_path = FIGURE_DIR / "initial_dataset_subcarrier_diversity.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    config = DatasetConfig()
    dataset = generate_dataset(config)
    dataset_path, metadata_path, summary = save_dataset(dataset, config)
    overview_path = save_overview_figure(dataset, config)
    subcarrier_path = save_subcarrier_figure(dataset, config)

    print("Initial CSI dataset generated and validated.")
    print(f"CSI shape: {tuple(summary['csi_shape'])}")
    print(f"CSI dtype: {summary['csi_dtype']}")
    print(f"All finite: {summary['all_finite']}")
    print(f"Dataset: {dataset_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Overview: {overview_path}")
    print(f"Subcarrier figure: {subcarrier_path}")
    print(f"SHA256: {summary['sha256']}")


if __name__ == "__main__":
    main()
